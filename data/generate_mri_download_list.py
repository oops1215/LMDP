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
        if "repeat" in d or "mpr-r" in d:
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

    # ── 第一步：绝对排除（无论何种描述均不保留）─────────────────────────────
    # repeat/-R;  → 重扫版本（MPR-R;...）
    # SENS        → Sensitivity 扫描变体
    # 非 T1 模态  → fMRI / DTI / DWI / BOLD / pcASL / FLAIR / SWI / T2 等
    # 衍生产品    → HarP / Reoriented / Brain Mask / MUSE（非原始结构像）
    # 注意：不用 _2\b，因为 "Scaled_2" 是 ADNI1 正常扫描的描述后缀，不应排除
    EXCLUDE = (
        r"(?i)(repeat\b|MPR-R|\bSENS\b"
        r"|fmri|dti|dwi|bold|pcasl|asl\b|flair|swi|t2\b"
        r"|\bharp\b|reoriented|brain[\s_]?mask|\bmuse\b)"
    )
    is_unwanted = df["SEQUENCE"].str.contains(EXCLUDE, regex=True, na=False)
    n_before = len(df)
    df = df[~is_unwanted].copy()
    print(f"  绝对排除后: {n_before} → {len(df)}（排除 {n_before - len(df)} 条）")

    # ── 第二步：MPR 型扫描（ADNI1/2/GO）必须同时含 N3 和 Scaled ──────────────
    # 目标描述：MPR; GradWarp; B1 Correction; N3; Scaled（或 Scaled_2 作备用）
    # 无 N3 或无 Scaled 的 MPR 变体跳过；缺失的其他预处理步骤可在脚本中补充
    # ADNI3/4 描述（如 "ADNI Brain T1 3T"）不含 MPR，不受此规则约束
    is_mpr = df["SEQUENCE"].str.contains(r"(?i)\bMPR\b", regex=True, na=False)
    mpr_no_n3     = is_mpr & ~df["SEQUENCE"].str.contains(r"(?i)\bN3\b",     regex=True, na=False)
    mpr_no_scaled = is_mpr & ~df["SEQUENCE"].str.contains(r"(?i)\bScaled\b", regex=True, na=False)
    is_incomplete_mpr = mpr_no_n3 | mpr_no_scaled
    n_before = len(df)
    df = df[~is_incomplete_mpr].copy()
    print(f"  MPR N3/Scaled 过滤后: {n_before} → {len(df)}（排除 {n_before - len(df)} 条缺 N3/Scaled 的 MPR）")

    # 统计最终序列分布
    print(f"  保留的序列分布:")
    print(df["SEQUENCE"].value_counts().head(10).to_string())
    return df


def _fallback_repeat_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    当 MRIMETA 不可用时，利用 PREFERRED_DESC（来自 UCSFFSX 文件名推断）
    做简单的关键字过滤。只能过滤掉明确含 repeat/-R/SENS/衍生产品关键字的记录，
    无法对无描述信息的 ID 做判断。
    """
    if "PREFERRED_DESC" not in df.columns:
        return df
    EXCLUDE = r"(?i)(repeat\b|MPR-R|\bSENS\b|\bharp\b|reoriented|brain[\s_]?mask|\bmuse\b)"
    is_unwanted = df["PREFERRED_DESC"].str.contains(EXCLUDE, regex=True, na=False)
    n_bad = is_unwanted.sum()
    if n_bad:
        print(f"  回退过滤: 排除 {n_bad} 条含 Repeat/SENS/衍生产品关键字的记录")
    return df[~is_unwanted].copy()


def add_ptid(df: pd.DataFrame, rda_dir: str) -> pd.DataFrame:
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


def generate_download_list(rda_dir: str, output_path: str) -> None:
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
    args = parser.parse_args()
    generate_download_list(args.rda_dir, args.output)
