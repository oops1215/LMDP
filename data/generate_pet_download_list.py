"""
从 ADNIMERGE2 PET 分析表提取 IMAGEUID，生成 FDG-PET 精确下载清单
=================================================================

与 MRI 不同，PET 没有像 UCSFFSX 那样把 IMAGEUID 固定到特定分析结果的文件。
本脚本从 UC Berkeley FDG-PET 分析表（UCBERKELEYFDG*.rda）中提取 IMAGEUID，
只保留与 MRI 下载清单中 PTID+VISCODE 完全一致的访视。

PET 目标版本（ADNI IDA 描述）：
  FDG  +  Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res
  （已与 MRI 配准，可跳过 PET→MRI 配准步骤）

用法：
  python data/generate_pet_download_list.py \\
      --rda_dir   ~/LMDP/ADNIMERGE2/data \\
      --mri_list  data/mri_download_list.csv \\
      --output    data/pet_download_list.csv

输出：
  data/pet_download_list.csv
  data/pet_download_list_ids_only.txt
  data/pet_download_list_ids_comma.txt
"""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def read_rda_safe(path: str):
    try:
        import pyreadr
    except ImportError:
        print("pip install pyreadr"); sys.exit(1)
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
    if vc in ("bl", "sc", "scmri", "init", "nv", "v1", "bas", "m0", "00m", "baseline"):
        return "bl"
    m = re.match(r"m?(\d+)m?$", vc)
    if m:
        n = int(m.group(1))
        return f"m{n:02d}" if n < 10 else f"m{n}"
    return vc


# ── 从 PET 分析表提取 IMAGEUID ─────────────────────────────────────────────

def extract_pet_imageids(rda_dir: str) -> pd.DataFrame:
    """
    从 UCBERKELEYFDG*.rda 等表中提取 RID, VISCODE, IMAGEUID。
    同时尝试 idaSearch_pet*.csv（IDA PET 搜索导出）作为补充。
    """
    # ── 1. rda 文件（UCBERKELEYFDG 系列）────────────────────────────────────
    pet_files = [
        "UCBERKELEYFDG.rda",
        "UCBERKELEYFDG6.rda",
        "UCBERKELEYFDG7.rda",
        "UCBERKELEYFDG_ADNI1.rda",
        "BAIPETNMRC_UCSFFSL.rda",
        "BAIPETNMRC_UCSFFSL6.rda",
    ]

    frames = []
    for fname in pet_files:
        path = os.path.join(rda_dir, fname)
        df = read_rda_safe(path)
        if df is None:
            continue

        # 找 IMAGEUID 列
        id_col = next((c for c in ["IMAGEUID", "IMAGE_ID", "PETIMAGEUID", "FDGIMAGEUID"]
                       if c in df.columns), None)
        if id_col is None:
            print(f"  {fname}: 无 IMAGEUID 列，跳过（列名: {list(df.columns[:10])}）")
            continue

        vc_col = "VISCODE2" if "VISCODE2" in df.columns else "VISCODE"
        sub = pd.DataFrame()
        sub["RID"]      = df["RID"]
        sub["VISCODE"]  = df[vc_col].apply(standardize_viscode)
        sub["IMAGEUID"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
        if "PTID" in df.columns:
            sub["PTID"] = df["PTID"]
        if "EXAMDATE" in df.columns:
            sub["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce").dt.strftime("%Y-%m-%d")
        sub["SOURCE"] = fname

        sub = sub[sub["IMAGEUID"].notna() & (sub["IMAGEUID"] > 0)]
        if len(sub) == 0:
            print(f"  {fname}: 无有效 IMAGEUID")
            continue

        frames.append(sub)
        print(f"  {fname}: {len(sub)} 条有效 IMAGEUID")

    # ── 2. IDA PET 搜索导出 CSV（idaSearch_pet*.csv 或 petSearch*.csv）────────
    csv_patterns = ["idaSearch_pet*.csv", "idaSearch*pet*.csv",
                    "petSearch*.csv", "PET_search*.csv"]
    for pattern in csv_patterns:
        for csv_path in sorted(glob.glob(os.path.join(rda_dir, pattern))):
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.upper().replace(" ", "_") for c in df.columns]
                id_col   = next((c for c in ["IMAGE_ID", "IMAGEUID"] if c in df.columns), None)
                desc_col = next((c for c in ["DESCRIPTION", "SEQUENCE"] if c in df.columns), None)
                if id_col is None:
                    continue
                # 只保留 FDG-PET（过滤掉 amyloid / tau 等）
                if desc_col:
                    is_fdg = df[desc_col].str.contains(r"(?i)fdg", na=False)
                    df = df[is_fdg]
                sub = pd.DataFrame()
                if "SUBJECT_ID" in df.columns:
                    sub["PTID"] = df["SUBJECT_ID"]
                if "VISIT" in df.columns:
                    sub["VISIT_RAW"] = df["VISIT"]
                sub["IMAGEUID"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
                sub["SOURCE"]   = os.path.basename(csv_path)
                sub = sub[sub["IMAGEUID"].notna() & (sub["IMAGEUID"] > 0)]
                if len(sub) > 0:
                    frames.append(sub)
                    print(f"  {os.path.basename(csv_path)}: {len(sub)} 条 FDG-PET IMAGEUID")
            except Exception as e:
                print(f"  [警告] {csv_path}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["RID", "VISCODE", "IMAGEUID"]
                                        if "RID" in combined.columns
                                        else ["IMAGEUID"])
    print(f"\n汇总: {len(combined)} 条 PET 候选记录")
    return combined


# ── 按 MRI 清单过滤，保留目标 PTID+VISCODE ───────────────────────────────────

def filter_by_mri_list(pet_df: pd.DataFrame, mri_list_path: str) -> pd.DataFrame:
    """只保留 MRI 下载清单里出现的 PTID+VISCODE 组合。"""
    if not os.path.exists(mri_list_path):
        print(f"  [警告] MRI 清单不存在: {mri_list_path}，跳过过滤")
        return pet_df

    mri = pd.read_csv(mri_list_path)
    mri["VISCODE"] = mri["VISCODE"].apply(standardize_viscode)

    # 用 RID+VISCODE 或 PTID+VISCODE 做 join
    if "RID" in pet_df.columns and "RID" in mri.columns:
        mri_keys = set(zip(mri["RID"].astype(str), mri["VISCODE"]))
        mask = pet_df.apply(
            lambda r: (str(r["RID"]), r["VISCODE"]) in mri_keys, axis=1)
    elif "PTID" in pet_df.columns and "PTID" in mri.columns:
        mri_keys = set(zip(mri["PTID"], mri["VISCODE"]))
        mask = pet_df.apply(
            lambda r: (r.get("PTID", ""), r["VISCODE"]) in mri_keys, axis=1)
    else:
        print("  [警告] 无法匹配 RID/PTID，返回全部 PET 记录")
        return pet_df

    filtered = pet_df[mask].copy()
    print(f"  按 MRI 清单过滤: {len(pet_df)} → {len(filtered)} 条")
    return filtered


# ── 每访视选最优 PET ──────────────────────────────────────────────────────────

def pick_best_pet_per_visit(df: pd.DataFrame) -> pd.DataFrame:
    """同一 RID/PTID+VISCODE 有多条时，保留最小 IMAGEUID（ADNI 惯例：最早处理）。"""
    key_cols = []
    if "RID" in df.columns:
        key_cols.append("RID")
    elif "PTID" in df.columns:
        key_cols.append("PTID")
    if "VISCODE" in df.columns:
        key_cols.append("VISCODE")

    if not key_cols:
        return df

    n_before = len(df)
    df = (df.sort_values(key_cols + ["IMAGEUID"])
            .drop_duplicates(subset=key_cols, keep="first")
            .reset_index(drop=True))
    if n_before > len(df):
        print(f"  去重: {n_before} → {len(df)}")
    return df


# ── 主函数 ────────────────────────────────────────────────────────────────────

def generate_pet_download_list(rda_dir: str, mri_list: str, output_path: str):
    rda_dir = os.path.expanduser(rda_dir)
    mri_list = os.path.expanduser(mri_list)

    print(f"\n从 {rda_dir} 提取 PET IMAGEUID ...")
    df = extract_pet_imageids(rda_dir)

    if df.empty:
        print("\n[失败] 未找到任何 PET IMAGEUID")
        print("""
未找到 UCBERKELEYFDG*.rda 或 PET CSV 文件。
请在 ADNI IDA Advanced Image Search 中手动搜索：
  Modality = PET
  Description = Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res
  Subject = （与 MRI 清单中的受试者相同）
然后将下载的 idaSearch CSV 命名为 idaSearch_pet.csv 放入 rda_dir 后重试。
""")
        return

    # 补充 PTID（若缺失）
    if "PTID" not in df.columns or df["PTID"].isna().mean() > 0.5:
        reg_path = os.path.join(rda_dir, "REGISTRY.rda")
        reg = read_rda_safe(reg_path)
        if reg is not None and "RID" in df.columns:
            if "PTID" not in reg.columns and "SITEID" in reg.columns:
                reg["PTID"] = (reg["SITEID"].astype(str).str.zfill(3) + "_S_" +
                               reg["RID"].astype(str).str.zfill(4))
            ptid_map = reg.drop_duplicates("RID").set_index("RID")["PTID"].to_dict()
            df["PTID"] = df["RID"].map(ptid_map)

    # 按 MRI 清单过滤
    if os.path.exists(mri_list):
        df = filter_by_mri_list(df, mri_list)
    else:
        print(f"  MRI 清单未找到（{mri_list}），保留全部访视")
        target = {"bl", "m12", "m24", "m36", "m48", "m60"}
        if "VISCODE" in df.columns:
            df = df[df["VISCODE"].isin(target)]

    df = pick_best_pet_per_visit(df)

    if df.empty:
        print("[失败] 过滤后无数据")
        return

    # 只保留目标访视
    target = {"bl", "m12", "m24", "m36", "m48", "m60"}
    if "VISCODE" in df.columns:
        df = df[df["VISCODE"].isin(target)].copy()

    print(f"\n目标访视 PET IMAGEUID: {len(df)} 条")
    if "VISCODE" in df.columns:
        print(df["VISCODE"].value_counts().sort_index().to_string())

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n保存至: {output_path}")

    id_list = df["IMAGEUID"].dropna().astype(int).tolist()
    id_file = output_path.replace(".csv", "_ids_only.txt")
    with open(id_file, "w") as f:
        f.write("\n".join(str(i) for i in id_list))
    id_file_csv = output_path.replace(".csv", "_ids_comma.txt")
    with open(id_file_csv, "w") as f:
        f.write(",".join(str(i) for i in id_list))

    print(f"纯 ID 列表（每行一个）: {id_file}")
    print(f"逗号分隔 ID（粘贴用）  : {id_file_csv}")
    print(f"共 {len(id_list)} 个 PET Image ID")
    print("""
=== ADNI IDA 下载说明 ===
1. 登录 ida.loni.usc.edu
2. Download → Image Collections → Advanced Image Search (beta)
3. 左侧 "Image ID" 栏粘贴 _ids_comma.txt 的内容
4. 全选 → Add to Collection → 下载 NIfTI
   预期返回图像数 ≈ ID 总数
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rda_dir",  required=True, help="ADNIMERGE2 的 data/ 目录")
    parser.add_argument("--mri_list", default="data/mri_download_list.csv",
                        help="MRI 下载清单（用于过滤匹配的 PTID+VISCODE）")
    parser.add_argument("--output",   default="data/pet_download_list.csv")
    args = parser.parse_args()
    generate_pet_download_list(args.rda_dir, args.mri_list, args.output)
