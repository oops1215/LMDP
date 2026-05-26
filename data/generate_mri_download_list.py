"""
从 UCSFFSX*.rda 提取 IMAGEUID，生成 MRI 精确下载清单
=====================================================

ADNI 的 FreeSurfer 脑区体积（Hippocampus、Ventricles 等）
是在特定 MRI 扫描上计算的，该扫描由 IMAGEUID 唯一标识。
只有下载这些对应的 MRI，图像特征才与表格生物标志物一致。

用法：
  python data/generate_mri_download_list.py \\
      --rda_dir ~/LMDP/ADNIMERGE2/data \\
      --output  data/mri_download_list.csv

输出格式（可直接在 ADNI IDA 中按 Image Data ID 搜索）：
  PTID, RID, VISCODE, EXAMDATE, IMAGEUID, FLDSTRENG
  002_S_0413, 413, bl, 2005-09-08, I45102, 1.5

在 ADNI IDA 中使用步骤：
  1. 登录 ida.loni.usc.edu
  2. Download → Image Collections → Advanced Image Search
  3. 搜索条件：Modality=MRI，Image ID = 上面 IMAGEUID 列的值
     （也可以直接在 Subject/Image ID 栏批量粘贴多个 ID）
  4. 添加到集合 → 下载 NIfTI
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_rda_safe(rda_dir: str, filename: str):
    try:
        import pyreadr
    except ImportError:
        print("pip install pyreadr"); sys.exit(1)
    path = os.path.join(rda_dir, filename)
    if not os.path.exists(path):
        return None
    result = pyreadr.read_r(path)
    df = result[list(result.keys())[0]]
    df.columns = [c.upper() for c in df.columns]
    return df


def standardize_viscode(vc):
    if not isinstance(vc, str):
        return ""
    import re
    vc = vc.strip().lower()
    if vc in ("bl","sc","scmri","init","nv","v1","bas","m0","00m","baseline"):
        return "bl"
    m = re.match(r"m?(\d+)m?$", vc)
    if m:
        n = int(m.group(1))
        return f"m{n:02d}" if n < 10 else f"m{n}"
    return vc


def extract_imageids_from_fs(rda_dir: str) -> pd.DataFrame:
    """
    从所有 UCSFFSX*.rda 文件提取 RID, VISCODE, IMAGEUID。
    每个 IMAGEUID 唯一对应 ADNI IDA 中的一张图像（FreeSurfer 处理时使用的那张）。
    在 IDA 中只需按 Image ID 搜索，不要加 Description 过滤——每个 ID 已唯一。
    """
    # 各来源文件对应的 ADNI 期别和典型图像描述（仅供参考，IDA 搜索不需要用）
    fs_files = [
        ("UCSFFSX.rda",           "MPR; GradWarp; B1 Correction; N3; Scaled"),      # ADNI1 1.5T
        ("UCSFFSX51.rda",         "MPR; GradWarp; B1 Correction; N3; Scaled"),      # ADNI1/GO/2
        ("UCSFFSX51ALL.rda",      "MPR; GradWarp; B1 Correction; N3; Scaled"),      # 部分版本别名
        ("UCSFFSX51_ADNI1_3T.rda","MPR; GradWarp; B1 Correction; N3; Scaled"),      # ADNI1 3T
        ("UCSFFSX6.rda",          "ADNI Brain T1 3T"),                              # ADNI3
        ("UCSFFSX7.rda",          "ADNI Brain T1 3T"),                              # ADNI4
    ]
    frames = []
    for fname, preferred_desc in fs_files:
        df = read_rda_safe(rda_dir, fname)
        if df is None:
            continue

        # 必需列
        if "IMAGEUID" not in df.columns:
            print(f"  {fname}: 没有 IMAGEUID 列，跳过")
            continue

        vc_col = "VISCODE2" if "VISCODE2" in df.columns else "VISCODE"
        sub = pd.DataFrame()
        sub["RID"]      = df["RID"]
        sub["VISCODE"]  = df[vc_col].apply(standardize_viscode)
        sub["IMAGEUID"] = pd.to_numeric(df["IMAGEUID"], errors="coerce").astype("Int64")

        if "EXAMDATE" in df.columns:
            sub["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "PTID" in df.columns:
            sub["PTID"] = df["PTID"]
        sub["PREFERRED_DESC"] = preferred_desc  # 该来源文件的典型图像描述
        sub["SOURCE"] = fname

        # QC 过滤
        qc = next((c for c in ["OVERALLQC","LHIPQC","HIPPOQC"] if c in df.columns), None)
        if qc:
            sub = sub[df[qc].astype(str).str.strip().str.upper().isin(
                ["PASS","1","1.0","TRUE"])]

        # 过滤无效 IMAGEUID
        sub = sub[sub["IMAGEUID"].notna() & (sub["IMAGEUID"] > 0)]

        if len(sub) == 0:
            print(f"  {fname}: 无有效 IMAGEUID（该文件可能不含图像元数据）")
            continue

        frames.append(sub)
        print(f"  {fname}: {len(sub)} 条有效 IMAGEUID  [{preferred_desc[:40]}...]")

    if not frames:
        print("[警告] 未从任何 UCSFFSX 文件中提取到 IMAGEUID")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # 同一 RID+VISCODE 可能有多条（ADNI 对同一访视处理了多个扫描）
    # 按描述优先级排序：standard Scaled > Scaled_2 > 未知 > SENS > repeat
    # 这样 drop_duplicates(keep="first") 保留的永远是最接近 primary scan 的那条
    def _scan_priority(desc: str) -> int:
        d = str(desc).lower()
        if "repeat" in d or "-r;" in d:
            return 10   # repeat — 最不优先
        if "sens" in d:
            return 8    # sensitivity variant
        if "scaled_2" in d or "scaled 2" in d:
            return 1    # secondary scaling pass，次优
        if "scaled" in d:
            return 0    # standard Scaled — 最优先
        return 5        # 其他未知序列

    combined["_priority"] = combined["PREFERRED_DESC"].apply(_scan_priority)
    combined = (combined
                .sort_values(["RID", "VISCODE", "_priority", "IMAGEUID"])
                .drop_duplicates(subset=["RID", "VISCODE"], keep="first")
                .drop(columns=["_priority"]))
    combined = combined.sort_values(["RID","VISCODE"]).reset_index(drop=True)
    print(f"\n汇总: {len(combined)} 条记录，{combined['RID'].nunique()} 名受试者")
    return combined


def filter_by_description(df: pd.DataFrame, rda_dir: str) -> pd.DataFrame:
    """
    用 MRIMETA.rda / MRI3META.rda 为每个 IMAGEUID 补充序列描述，
    过滤掉 Repeat 扫描和非目标序列。

    ADNI 命名规则：
      - 正常：MPR; GradWarp; B1 Correction; N3; Scaled 或 Scaled_2（均为正常 ADNI1 扫描）
      - Repeat：MPR-R; GradWarp... 或 SEQUENCE 含 REPEAT（关键词）
      - Sensitivity：MPR; ; N3; Scaled 或含 SENS 关键词
    """
    # 合并 MRIMETA（ADNI1/2/GO）和 MRI3META（ADNI3/4）
    frames = []
    for fname in ["MRIMETA.rda", "MRI3META.rda"]:
        path = os.path.join(rda_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            import pyreadr
            result = pyreadr.read_r(path)
            mri = result[list(result.keys())[0]]
            mri.columns = [c.upper() for c in mri.columns]
            frames.append(mri)
            print(f"  读取 {fname}: {len(mri)} 行")
        except Exception as e:
            print(f"  [警告] 读取 {fname} 失败: {e}")

    if not frames:
        print("  [提示] 未找到 MRIMETA.rda / MRI3META.rda，改用关键字回退过滤")
        return _fallback_repeat_filter(df)

    meta = pd.concat(frames, ignore_index=True)

    # 找 Image ID 列
    id_col = next((c for c in ["IMAGEUID", "IMAGE_ID", "IMAGEID"] if c in meta.columns), None)
    # 找描述列（MRIMETA 通常用 SEQUENCE 或 MRITYPE 或 SERIESTYPE）
    desc_col = next((c for c in ["SEQUENCE", "MRITYPE", "SERIESTYPE",
                                  "DESCRIPTION", "IMAGEDESC"] if c in meta.columns), None)
    if id_col is None or desc_col is None:
        print(f"  [警告] MRIMETA 缺少 Image ID 或描述列，改用关键字回退过滤")
        print(f"         实际列名: {list(meta.columns[:20])}")
        return _fallback_repeat_filter(df)

    print(f"  使用列: {id_col} → {desc_col}")
    print(f"  描述样本: {meta[desc_col].dropna().unique()[:8].tolist()}")

    meta["IMAGEUID_INT"] = pd.to_numeric(meta[id_col], errors="coerce").astype("Int64")
    desc_map = (meta.dropna(subset=["IMAGEUID_INT"])
                .set_index("IMAGEUID_INT")[desc_col]
                .to_dict())

    df = df.copy()
    df["SEQUENCE"] = df["IMAGEUID"].map(desc_map).fillna("")

    # 过滤 Repeat / Sensitivity / 非 T1 结构像
    # 注意：不用 _2\b，因为 "Scaled_2" 是 ADNI1 正常扫描的描述后缀
    # "MPRAGE REPEAT" / "MPR-R;" 才是 repeat 标志；"SENS" / "MPRAGE SENS" 是 sensitivity
    EXCLUDE = r"(?i)(repeat\b|-R;|\bSENS\b|fmri|dti|dwi|bold|pcasl|asl\b|flair|swi|t2\b)"
    is_unwanted = df["SEQUENCE"].str.contains(EXCLUDE, regex=True, na=False)
    n_before = len(df)
    df = df[~is_unwanted].copy()
    print(f"  过滤前: {n_before}  →  过滤后: {len(df)}（排除 {n_before - len(df)} 条 Repeat/非T1）")

    # 统计最终序列分布
    print(f"  保留的序列分布:")
    print(df["SEQUENCE"].value_counts().head(10).to_string())
    return df


def _fallback_repeat_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    当 MRIMETA 不可用时，利用 PREFERRED_DESC（来自 UCSFFSX 文件名推断）
    做简单的关键字过滤。只能过滤掉明确含 repeat/-R/SENS 关键字的 SEQUENCE，
    无法对无描述信息的 ID 做判断。
    """
    if "PREFERRED_DESC" not in df.columns:
        return df
    EXCLUDE = r"(?i)(repeat\b|-R;|\bSENS\b)"
    is_unwanted = df["PREFERRED_DESC"].str.contains(EXCLUDE, regex=True, na=False)
    n_bad = is_unwanted.sum()
    if n_bad:
        print(f"  回退过滤: 排除 {n_bad} 条含 Repeat/SENS 关键字的记录")
    return df[~is_unwanted].copy()


def filter_by_ida_csv(df: pd.DataFrame, ida_csv_path: str) -> pd.DataFrame:
    """
    利用 IDA "Advanced Search Results" 导出的 CSV 做交叉验证。

    该 CSV（CSV Download 按钮导出）包含列：
      Subject ID, Phase, Sex, Research Group, Visit, Age, Modality, Description
    但 **没有** Image ID（IMAGEUID）列。

    策略：
    - 对每个 (PTID, VISCODE) 查看 IDA CSV 里对应的所有 Description
    - 若存在至少一条非 repeat / 非 SENS 的描述 → 认为 UCSFFSX 里的 IMAGEUID
      是正常 scan，保留
    - 若只有 repeat/SENS 描述 → IMAGEUID 对应 repeat scan（FreeSurfer 被迫使用），
      打印警告并保留（无替代 scan，仍需下载）
    - 若 (PTID, VISCODE) 在 IDA CSV 里完全找不到 → 无法判断，保留并警告
    """
    # IDA Visit 名称 → VISCODE 的映射（近似）
    VISIT_MAP = {
        "ADNI Screening":              "bl",
        "ADNI Baseline":               "bl",
        "ADNIGO Screening MRI":        "bl",
        "ADNI2 Screening MRI-New Pt":  "bl",
        "ADNI2 Initial Visit-Cont Pt": "bl",
        "ADNI1/GO Month 12":           "m12",
        "ADNI2 Year 1 Visit":          "m12",
        "ADNI1/GO Month 24":           "m24",
        "ADNI2 Year 2 Visit":          "m24",
        "ADNI1/GO Month 36":           "m36",
        "ADNI2 Year 3 Visit":          "m36",
        "ADNI1/GO Month 48":           "m48",
        "ADNI2 Year 4 Visit":          "m48",
        "ADNIGO Month 60":             "m60",
        "ADNI2 Year 5 Visit":          "m60",
        "ADNI3 Initial Visit-Cont Pt": "bl",
        "ADNI3 Year 1 Visit":          "m12",
        "ADNI3 Year 2 Visit":          "m24",
        "ADNI4 Initial Visit-Cont Pt": "bl",
    }
    REPEAT_PAT = r"(?i)(repeat\b|-R;)"
    SENS_PAT   = r"(?i)\bSENS\b"

    try:
        ida = pd.read_csv(ida_csv_path)
    except Exception as e:
        print(f"  [警告] 无法读取 IDA CSV: {e}，跳过交叉验证")
        return df

    # 标准化列名
    ida.columns = [c.strip() for c in ida.columns]
    if "Subject ID" not in ida.columns or "Visit" not in ida.columns or "Description" not in ida.columns:
        print(f"  [警告] IDA CSV 缺少必要列（Subject ID / Visit / Description），跳过")
        return df

    ida = ida[ida["Modality"].astype(str).str.upper() == "MRI"].copy() if "Modality" in ida.columns else ida.copy()
    ida["VISCODE"] = ida["Visit"].map(VISIT_MAP)
    ida = ida.dropna(subset=["VISCODE"])
    ida["is_bad"] = (ida["Description"].str.contains(REPEAT_PAT, regex=True, na=False) |
                     ida["Description"].str.contains(SENS_PAT,   regex=True, na=False))

    # 对每个 (PTID, VISCODE) 判断：是否存在 good scan
    good_set = set(
        ida[~ida["is_bad"]][["Subject ID","VISCODE"]]
        .itertuples(index=False, name=None)
    )
    bad_only_set = set(
        ida.groupby(["Subject ID","VISCODE"])
        .filter(lambda g: g["is_bad"].all())[["Subject ID","VISCODE"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    if "PTID" not in df.columns:
        print("  [提示] 下载清单缺少 PTID 列，跳过 IDA CSV 交叉验证")
        return df

    repeat_ids = []
    for _, row in df.iterrows():
        key = (row["PTID"], row["VISCODE"])
        if key in bad_only_set:
            repeat_ids.append(int(row["IMAGEUID"]))

    if repeat_ids:
        print(f"\n[IDA CSV 交叉验证]")
        print(f"  发现 {len(repeat_ids)} 个 IMAGEUID 在 IDA 里只对应 repeat/SENS 扫描：")
        print(f"  {repeat_ids[:20]}{'...' if len(repeat_ids) > 20 else ''}")
        print(f"  这些 scan 是 FreeSurfer 被迫使用 repeat 的情况（原始 scan 不可用）。")
        print(f"  已保留在下载清单中（无替代），请在分析时酌情排除。")
        # 在 CSV 里加标记列方便后续筛选
        df = df.copy()
        df["IS_REPEAT_ONLY"] = df["IMAGEUID"].isin(repeat_ids).astype(int)
    else:
        print(f"\n[IDA CSV 交叉验证] 未发现 repeat-only IMAGEUID，下载清单干净。")
        df = df.copy()
        df["IS_REPEAT_ONLY"] = 0

    not_found = df[~df["PTID"].apply(lambda p: any(p == k[0] for k in good_set | bad_only_set))]
    if len(not_found):
        print(f"  另有 {len(not_found)} 条记录在 IDA CSV 里找不到对应访视（可能 Visit 名称未映射）")

    return df
    """补充 PTID（若 UCSFFSX 没有，从 REGISTRY 获取）。"""
    if "PTID" in df.columns and df["PTID"].notna().mean() > 0.5:
        return df
    reg = read_rda_safe(rda_dir, "REGISTRY.rda")
    if reg is None:
        return df
    reg.columns = [c.upper() for c in reg.columns]
    if "PTID" not in reg.columns:
        # 用 SITEID + RID 构造
        if "SITEID" in reg.columns:
            reg["PTID"] = (reg["SITEID"].astype(str).str.zfill(3) + "_S_" +
                           reg["RID"].astype(str).str.zfill(4))
    ptid_map = reg.drop_duplicates("RID")[["RID","PTID"]]
    df = df.merge(ptid_map, on="RID", how="left", suffixes=("","_reg"))
    if "PTID_reg" in df.columns:
        df["PTID"] = df["PTID"].fillna(df["PTID_reg"])
        df = df.drop(columns=["PTID_reg"])
    return df


def generate_download_list(rda_dir: str, output_path: str,
                           ida_csv: str | None = None) -> None:
    rda_dir = os.path.expanduser(rda_dir)

    print(f"\n从 {rda_dir} 提取 IMAGEUID ...")
    df = extract_imageids_from_fs(rda_dir)
    if df.empty:
        print("[失败] 未找到有效数据")
        return

    df = filter_by_description(df, rda_dir)
    df = add_ptid(df, rda_dir)

    # 只保留目标访视（论文用 bl/m12/m24/m36/m48/m60）
    target = {"bl","m12","m24","m36","m48","m60"}
    df_target = df[df["VISCODE"].isin(target)].copy()
    print(f"\n目标访视（bl/m12/m24/m36/m48/m60）: {len(df_target)} 条")

    # 用 IDA 搜索 CSV 做交叉验证（可选）
    if ida_csv:
        ida_csv = os.path.expanduser(ida_csv)
        print(f"\n用 IDA CSV 做交叉验证: {ida_csv}")
        df_target = filter_by_ida_csv(df_target, ida_csv)

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_target.to_csv(output_path, index=False)
    print(f"保存至: {output_path}")

    # 统计
    print(f"\n各访视 IMAGEUID 数量:")
    print(df_target["VISCODE"].value_counts().sort_index().to_string())
    print(f"\n各来源文件分布:")
    print(df_target["SOURCE"].value_counts().to_string())

    # 生成 ADNI IDA 搜索用的 Image ID 列表（两种格式）
    id_list = df_target["IMAGEUID"].dropna().astype(int).tolist()
    id_file = output_path.replace(".csv", "_ids_only.txt")
    with open(id_file, "w") as f:
        f.write("\n".join(str(i) for i in id_list))
    id_file_csv = output_path.replace(".csv", "_ids_comma.txt")
    with open(id_file_csv, "w") as f:
        f.write(",".join(str(i) for i in id_list))
    print(f"\n纯 ID 列表（每行一个）: {id_file}")
    print(f"逗号分隔 ID（粘贴用）  : {id_file_csv}")
    print(f"共 {len(id_list)} 个 Image ID")
    print("""
=== ADNI IDA 下载说明 ===
1. 登录 ida.loni.usc.edu
2. Download → Image Collections → Advanced Image Search (beta)
3. 左侧 "Image ID" 栏粘贴 _ids_comma.txt 的内容
   ⚠️  不要加任何 Image Description / Modality 过滤
      每个 Image ID 唯一对应一张图，加过滤只会减少结果
4. 全选 → Add to Collection → 下载 NIfTI
   预期返回图像数 ≈ ID 总数（部分因权限不足可能缺失）
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rda_dir", required=True,
                        help="ADNIMERGE2 的 data/ 目录")
    parser.add_argument("--output",  default="data/mri_download_list.csv")
    parser.add_argument("--ida_csv", default=None,
                        help="IDA Advanced Search 导出的 CSV（可选），用于交叉验证 repeat scan")
    args = parser.parse_args()
    generate_download_list(args.rda_dir, args.output, ida_csv=args.ida_csv)
