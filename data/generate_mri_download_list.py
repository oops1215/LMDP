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

    # 去除无效 IMAGEUID（QC 过滤已在各分支完成，这里只去重完全相同的行）
    combined = combined.drop_duplicates(subset=["RID", "VISCODE", "IMAGEUID"])
    combined = combined.sort_values(["RID", "VISCODE", "IMAGEUID"]).reset_index(drop=True)
    print(f"\n汇总: {len(combined)} 条候选记录，{combined['RID'].nunique()} 名受试者")
    # 注意：此时同一 RID+VISCODE 可能有多条（来自不同 rda 文件或同一文件多次扫描）
    # 最终去重在 pick_best_scan_per_visit() 中基于真实 SEQUENCE 完成
    return combined


def filter_by_description(df: pd.DataFrame, rda_dir: str) -> pd.DataFrame:
    """
    用图像级元数据为每个 IMAGEUID 补充序列描述，过滤掉 Repeat 扫描和非目标序列。

    元数据来源（按优先顺序）：
      1. idaSearch*.csv  — ADNI IDA Advanced Search 导出文件（含 Image ID + Description）
      2. MRILIST.rda / MRI3LIST.rda — ADNI 图像列表（若有）
      3. MRIMETA.rda / MRI3META.rda — 协议参数表（通常无 IMAGEUID，会触发回退）

    ADNI 命名规则：
      - 正常：MPR; GradWarp; B1 Correction; N3; Scaled 或 Scaled_2
      - Repeat：MPR-R; ... / MPRAGE REPEAT / MPRAGE_ASO_repeat（均含 MPR-R 或 repeat）
      - Sensitivity：含 SENS 关键词
    """
    desc_map = _build_desc_map(rda_dir)
    if not desc_map:
        print("  [提示] 未获取到图像描述，改用关键字回退过滤")
        print("         建议将 ADNI IDA 导出的 idaSearch*.csv 放入 rda_dir 后重试")
        return _fallback_repeat_filter(df)

    df = df.copy()
    df["SEQUENCE"] = df["IMAGEUID"].map(desc_map).fillna("")
    n_matched = (df["SEQUENCE"] != "").sum()
    print(f"  SEQUENCE 匹配率: {n_matched}/{len(df)} ({100*n_matched/len(df):.1f}%)")

    return _apply_sequence_filters(df)


def _build_desc_map(rda_dir: str) -> dict:
    """
    构建 IMAGEUID → 序列描述 字典。
    优先 idaSearch*.csv，其次 rda 图像列表文件。
    """
    import glob

    # ── 优先：IDA 搜索导出 CSV（idaSearch*.csv）───────────────────────────
    ida_csvs = sorted(glob.glob(os.path.join(rda_dir, "idaSearch*.csv")))
    if ida_csvs:
        frames = []
        for path in ida_csvs:
            try:
                df = pd.read_csv(path)
                # 列名统一：大写 + 空格→下划线，"Image ID" → "IMAGE_ID"
                df.columns = [c.upper().replace(" ", "_") for c in df.columns]
                frames.append(df)
                print(f"  读取 {os.path.basename(path)}: {len(df)} 行")
            except Exception as e:
                print(f"  [警告] 读取 {path} 失败: {e}")
        if frames:
            meta = pd.concat(frames, ignore_index=True)
            id_col   = next((c for c in ["IMAGE_ID", "IMAGEUID", "IMAGEID"] if c in meta.columns), None)
            desc_col = next((c for c in ["DESCRIPTION", "SEQUENCE"] if c in meta.columns), None)
            if id_col and desc_col:
                meta["_ID"] = pd.to_numeric(meta[id_col], errors="coerce").astype("Int64")
                desc_map = (meta.dropna(subset=["_ID"])
                            .set_index("_ID")[desc_col]
                            .to_dict())
                print(f"  IDA CSV: {len(desc_map)} 条 IMAGE_ID → Description 映射")
                return desc_map
            print(f"  [警告] IDA CSV 缺少 Image ID 或 Description 列: {list(meta.columns[:10])}")

    # ── 备选：rda 图像列表文件 ─────────────────────────────────────────────
    frames = []
    for fname in ["MRILIST.rda", "MRI3LIST.rda", "MRIMETA.rda", "MRI3META.rda"]:
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
        return {}

    meta = pd.concat(frames, ignore_index=True)
    id_col   = next((c for c in ["IMAGE_ID", "IMAGEUID", "IMAGEID", "IMAGE_DATA_ID"]
                     if c in meta.columns), None)
    desc_col = next((c for c in ["DESCRIPTION", "SEQUENCE", "MRITYPE", "SERIESTYPE",
                                  "IMAGEDESC", "SERIES_DESCRIPTION"] if c in meta.columns), None)
    if id_col is None or desc_col is None:
        print(f"  [警告] rda 文件缺少 Image ID 或描述列（实际列名: {list(meta.columns[:15])}）")
        return {}

    print(f"  使用列: {id_col} → {desc_col}")
    meta["_ID"] = pd.to_numeric(meta[id_col], errors="coerce").astype("Int64")
    return (meta.dropna(subset=["_ID"])
            .set_index("_ID")[desc_col]
            .to_dict())


def _apply_sequence_filters(df: pd.DataFrame) -> pd.DataFrame:
    """在 df['SEQUENCE'] 上执行两阶段过滤（绝对排除 + MPR N3/Scaled 必要条件）。"""
    # ── 第一步：绝对排除 ──────────────────────────────────────────────────
    # repeat / MPR-R → 重扫版本
    # SENS           → Sensitivity 扫描变体
    # 非 T1 模态     → fMRI / DTI / DWI / BOLD / pcASL / FLAIR / SWI / T2
    # 衍生产品       → HarP / Reoriented / Brain Mask / MUSE
    EXCLUDE = (
        r"(?i)(?:repeat|MPR-R|\bSENS\b"
        r"|fmri|dti|dwi|bold|pcasl|asl\b|flair|swi|t2\b"
        r"|\bharp\b|reoriented|brain[\s_]?mask|\bmuse\b)"
    )
    is_unwanted = df["SEQUENCE"].str.contains(EXCLUDE, regex=True, na=False)
    n_before = len(df)
    df = df[~is_unwanted].copy()
    print(f"  绝对排除后: {n_before} → {len(df)}（排除 {n_before - len(df)} 条）")

    # ── 第二步：MPR 型扫描必须同时含 N3 和 Scaled ─────────────────────────
    # ADNI3/4（MT1; GradWarp; N3m 等格式）不含 MPR，不受此规则约束
    # 缺失的 GradWarp / B1 等步骤可在预处理脚本中补充
    is_mpr        = df["SEQUENCE"].str.contains(r"(?i)\bMPR\b",    regex=True, na=False)
    mpr_no_n3     = is_mpr & ~df["SEQUENCE"].str.contains(r"(?i)\bN3\b",     regex=True, na=False)
    mpr_no_scaled = is_mpr & ~df["SEQUENCE"].str.contains(r"(?i)\bScaled\b", regex=True, na=False)
    n_before = len(df)
    df = df[~(mpr_no_n3 | mpr_no_scaled)].copy()
    print(f"  MPR N3/Scaled 过滤后: {n_before} → {len(df)}（排除 {n_before - len(df)} 条）")

    print("  保留的序列分布:")
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
    EXCLUDE = r"(?i)(?:repeat|MPR-R|\bSENS\b|\bharp\b|reoriented|brain[\s_]?mask|\bmuse\b)"
    is_unwanted = df["PREFERRED_DESC"].str.contains(EXCLUDE, regex=True, na=False)
    n_bad = is_unwanted.sum()
    if n_bad:
        print(f"  回退过滤: 排除 {n_bad} 条含 Repeat/SENS/衍生产品关键字的记录")
    return df[~is_unwanted].copy()


def _sequence_priority(seq: str) -> int:
    s = str(seq).lower()
    if "repeat" in s or "mpr-r" in s:
        return 10  # repeat/rescan — 最不优先
    if "sens" in s:
        return 8
    if "scaled_2" in s or "scaled 2" in s:
        return 1   # Scaled_2 备用
    if "scaled" in s:
        return 0   # standard Scaled — 最优先
    return 5


def pick_best_scan_per_visit(df: pd.DataFrame) -> pd.DataFrame:
    """
    同一 RID+VISCODE 若有多条候选，按真实 SEQUENCE 优先级选最优一条。
    优先级：Scaled(0) > Scaled_2(1) > 未知(5) > SENS(8) > repeat/MPR-R(10)

    此函数在 filter_by_description 之后调用，此时 SEQUENCE 已由 MRIMETA 填充。
    若 SEQUENCE 列不存在（回退路径），则直接按 IMAGEUID 升序保留第一条。
    """
    col = "SEQUENCE" if "SEQUENCE" in df.columns else None
    df = df.copy()
    if col:
        df["_pri"] = df[col].apply(_sequence_priority)
    else:
        df["_pri"] = 5

    n_before = len(df)
    df = (df.sort_values(["RID", "VISCODE", "_pri", "IMAGEUID"])
            .drop_duplicates(subset=["RID", "VISCODE"], keep="first")
            .drop(columns=["_pri"]))
    df = df.sort_values(["RID", "VISCODE"]).reset_index(drop=True)
    removed = n_before - len(df)
    if removed:
        print(f"  pick_best_scan_per_visit: {n_before} → {len(df)}（丢弃 {removed} 条次优/重复候选）")
    return df


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
    df = pick_best_scan_per_visit(df)
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
