"""
从 IDA 导出的 PET 搜索 CSV 中提取 FDG-PET IMAGEUID，
与 MRI 下载清单的 Subject ID + 访视匹配，生成 PET 精确下载清单
=================================================================

步骤：
  1. 登录 ida.loni.usc.edu
  2. Download → Image Collections → Advanced Image Search (beta)
  3. 搜索条件（不限 Subject，搜全部）：
       Modality  = PET
       ──在 Description 栏输入──
       Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res
  4. Select All → 导出 CSV（点击 "CSV"按钮，不要 Add to Collection）
     ⚠️  导出的是搜索结果 CSV，文件名类似 idaSearch_xx_xx_xxxx.csv
  5. 把该文件命名为 idaSearch_pet.csv 放到 --rda_dir 目录下
  6. 运行本脚本，自动按 MRI 清单的 Subject ID + 访视过滤

用法：
  python data/generate_pet_download_list.py \\
      --rda_dir   ~/LMDP/ADNIMERGE2/data \\
      --mri_list  data/mri_download_list.csv \\
      --output    data/pet_download_list.csv

备选：若已有 UCBERKELEYFDG*.rda 也会自动尝试。
"""

import os
import sys
import glob
import argparse
import re
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── 访视标签 → VISCODE 映射 ────────────────────────────────────────────────

# ADNI IDA 的 Visit 字段 → 统一 VISCODE
_VISIT_MAP = {
    # baseline 类
    "adni screening":                   "bl",
    "adni baseline":                    "bl",
    "adni1/go baseline":                "bl",
    "adni2 baseline-new pt":            "bl",
    "adni2 initial visit-cont pt":      "bl",
    "adni3 initial visit-cont pt":      "bl",
    "adni4 baseline - new pt":          "bl",
    "adni4 initial visit - cont pt":    "bl",
    "no visit defined":                 "bl",
    # month 系列
    "adni1/go month 6":     "m06",
    "adni1/go month 12":    "m12",
    "adni1/go month 18":    "m18",
    "adni1/go month 24":    "m24",
    "adni1/go month 36":    "m36",
    "adni1/go month 48":    "m48",
    "adnigo month 60":      "m60",
    "adni2 year 1 visit":   "m12",
    "adni2 year 2 visit":   "m24",
    "adni2 year 3 visit":   "m36",
    "adni2 year 4 visit":   "m48",
    "adni2 year 5 visit":   "m60",
    "adni3 year 1 visit":   "m12",
    "adni3 year 2 visit":   "m24",
    "adni3 year 3 visit":   "m36",
    "adni3 year 4 visit":   "m48",
    "adni3 year 5 visit":   "m60",
    "adni4 month 12":       "m12",
    "month 24":             "m24",
    "month 36":             "m36",
}

def visit_to_viscode(visit_str: str) -> str:
    if not isinstance(visit_str, str):
        return ""
    key = visit_str.strip().lower()
    if key in _VISIT_MAP:
        return _VISIT_MAP[key]
    # 正则匹配 "Month N" / "Year N"
    m = re.search(r"month\s+(\d+)", key)
    if m:
        n = int(m.group(1))
        return f"m{n:02d}" if n < 10 else f"m{n}"
    m = re.search(r"year\s+(\d+)", key)
    if m:
        n = int(m.group(1)) * 12
        return f"m{n:02d}" if n < 10 else f"m{n}"
    return ""


def standardize_viscode(vc: str) -> str:
    if not isinstance(vc, str):
        return ""
    vc = vc.strip().lower()
    if vc in ("bl", "sc", "scmri", "init", "nv", "v1", "bas", "m0", "00m", "baseline"):
        return "bl"
    m = re.match(r"m?(\d+)m?$", vc)
    if m:
        n = int(m.group(1))
        return f"m{n:02d}" if n < 10 else f"m{n}"
    return vc


# ── 从 idaSearch_pet.csv 加载 FDG-PET ─────────────────────────────────────

def load_pet_from_ida_csv(rda_dir: str) -> pd.DataFrame:
    """
    读取 idaSearch_pet*.csv（IDA PET 搜索导出）。
    列名：Subject ID, Visit, Description, Image ID（经 upper+下划线后）
    只保留 FDG-PET，过滤掉 amyloid/tau PET。
    """
    patterns = [
        "idaSearch_pet*.csv",
        "idaSearch_Pet*.csv",
        "idaSearch_PET*.csv",
        "petSearch*.csv",
        "PET_search*.csv",
        "idaSearch*PET*.csv",
        "idaSearch*pet*.csv",
    ]
    csvs = []
    for p in patterns:
        csvs.extend(glob.glob(os.path.join(rda_dir, p)))
    csvs = sorted(set(csvs))

    if not csvs:
        return pd.DataFrame()

    frames = []
    for path in csvs:
        try:
            df = pd.read_csv(path)
            df.columns = [c.upper().replace(" ", "_") for c in df.columns]
            frames.append(df)
            print(f"  读取 {os.path.basename(path)}: {len(df)} 行")
        except Exception as e:
            print(f"  [警告] {path}: {e}")

    if not frames:
        return pd.DataFrame()

    meta = pd.concat(frames, ignore_index=True)

    id_col   = next((c for c in ["IMAGE_ID", "IMAGEUID", "IMAGEID"] if c in meta.columns), None)
    desc_col = next((c for c in ["DESCRIPTION", "SEQUENCE"] if c in meta.columns), None)
    subj_col = next((c for c in ["SUBJECT_ID", "PTID", "SUBJECT"] if c in meta.columns), None)
    visit_col = next((c for c in ["VISIT", "VISCODE"] if c in meta.columns), None)

    if id_col is None:
        print(f"  [警告] CSV 无 Image ID 列，实际列名: {list(meta.columns[:10])}")
        return pd.DataFrame()

    # 只保留 FDG（排除 AV45/AV1451/FBB 等 amyloid/tau）
    if desc_col:
        is_fdg = meta[desc_col].str.contains(r"(?i)fdg", na=False)
        n_before = len(meta)
        meta = meta[is_fdg].copy()
        print(f"  FDG 过滤: {n_before} → {len(meta)} 条")

    # 排除 Dynamic 序列（动态采集，非标准静态图像）
    if desc_col:
        is_dynamic = meta[desc_col].str.contains(r"(?i)dynamic", na=False)
        meta = meta[~is_dynamic].copy()

    result = pd.DataFrame()
    result["IMAGEUID"] = pd.to_numeric(meta[id_col], errors="coerce").astype("Int64")
    if subj_col:
        result["PTID"] = meta[subj_col].astype(str)
    if visit_col:
        result["VISCODE"] = meta[visit_col].apply(visit_to_viscode)
    if desc_col:
        result["DESCRIPTION"] = meta[desc_col]
        # 描述优先级：Uniform 6mm Res(0) > Uniform Resolution(1) >
        #             Standardized Image and Voxel Size(2) > Co-reg Averaged(3)
        def _pet_priority(d: str) -> int:
            d = str(d).lower()
            if "uniform 6mm res" in d:   return 0
            if "uniform resolution" in d: return 1
            if "standardized image" in d: return 2
            if "co-registered, averaged" in d or "coreg, avg" in d: return 3
            return 9
        result["_PRI"] = result["DESCRIPTION"].apply(_pet_priority)
        print("  描述优先级分布:")
        for pri, grp in result.groupby("_PRI"):
            sample = grp["DESCRIPTION"].iloc[0].split("<-")[0].strip()
            print(f"    优先级{pri}: {len(grp):4d} 条  [{sample[:60]}]")

    result = result[result["IMAGEUID"].notna() & (result["IMAGEUID"] > 0)]
    return result


# ── 从 UCBERKELEYFDG*.rda 加载（备选）────────────────────────────────────

def load_pet_from_rda(rda_dir: str) -> pd.DataFrame:
    rda_files = [
        "UCBERKELEYFDG.rda", "UCBERKELEYFDG6.rda", "UCBERKELEYFDG7.rda",
        "UCBERKELEYFDG_ADNI1.rda", "BAIPETNMRC_UCSFFSL.rda",
    ]
    frames = []
    for fname in rda_files:
        path = os.path.join(rda_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            import pyreadr
            result = pyreadr.read_r(path)
            df = result[list(result.keys())[0]]
            df.columns = [c.upper() for c in df.columns]
        except Exception as e:
            print(f"  [警告] {fname}: {e}")
            continue

        id_col = next((c for c in ["IMAGEUID", "IMAGE_ID", "PETIMAGEUID"]
                       if c in df.columns), None)
        if id_col is None:
            continue

        vc_col = "VISCODE2" if "VISCODE2" in df.columns else "VISCODE"
        sub = pd.DataFrame()
        sub["RID"]      = df["RID"]
        sub["VISCODE"]  = df[vc_col].apply(standardize_viscode)
        sub["IMAGEUID"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
        if "PTID" in df.columns:
            sub["PTID"] = df["PTID"]
        sub = sub[sub["IMAGEUID"].notna() & (sub["IMAGEUID"] > 0)]
        if len(sub):
            frames.append(sub)
            print(f"  {fname}: {len(sub)} 条")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── 按 MRI 清单过滤 ────────────────────────────────────────────────────────

def filter_by_mri_list(pet_df: pd.DataFrame, mri_list_path: str) -> pd.DataFrame:
    """只保留 MRI 下载清单中出现的 PTID+VISCODE 组合。"""
    mri = pd.read_csv(mri_list_path)
    mri["VISCODE"] = mri["VISCODE"].apply(standardize_viscode)

    # 建目标集合
    if "PTID" in mri.columns and "PTID" in pet_df.columns:
        target = set(zip(mri["PTID"].astype(str), mri["VISCODE"]))
        mask = [( str(r.get("PTID","")), r.get("VISCODE","") ) in target
                for _, r in pet_df.iterrows()]
    elif "RID" in mri.columns and "RID" in pet_df.columns:
        target = set(zip(mri["RID"].astype(str), mri["VISCODE"]))
        mask = [( str(r.get("RID","")), r.get("VISCODE","") ) in target
                for _, r in pet_df.iterrows()]
    else:
        print("  [警告] 无法匹配 PTID/RID，跳过过滤")
        return pet_df

    filtered = pet_df[mask].copy()
    print(f"  按 MRI 清单过滤: {len(pet_df)} → {len(filtered)} 条")
    return filtered


# ── 主函数 ────────────────────────────────────────────────────────────────

def generate_pet_download_list(rda_dir: str, mri_list: str, output_path: str):
    rda_dir  = os.path.expanduser(rda_dir)
    mri_list = os.path.expanduser(mri_list)

    print(f"\n从 {rda_dir} 提取 FDG-PET IMAGEUID ...")

    # 优先 IDA CSV，其次 rda
    df = load_pet_from_ida_csv(rda_dir)
    if df.empty:
        print("  未找到 idaSearch_pet*.csv，尝试 UCBERKELEYFDG*.rda ...")
        df = load_pet_from_rda(rda_dir)

    if df.empty:
        print("""
[失败] 未找到 FDG-PET 数据。请按以下步骤操作：

1. 登录 ida.loni.usc.edu
2. Download → Image Collections → Advanced Image Search (beta)
3. 搜索条件（搜索全部受试者，不限 Subject）：
     Modality   = PET
     Description 输入框填写：
     Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res
4. 点击 Search → 结果页面点击 "CSV" 按钮导出搜索结果
   （不是 Add to Collection，是直接导出 CSV）
5. 将文件重命名为 idaSearch_pet.csv 放到 rda_dir 目录
6. 重新运行本脚本
""")
        return

    print(f"汇总: {len(df)} 条 FDG-PET 候选")

    # 过滤空 VISCODE
    if "VISCODE" in df.columns:
        n = len(df)
        df = df[df["VISCODE"] != ""].copy()
        if len(df) < n:
            print(f"  过滤无法识别访视的记录: {n} → {len(df)}")

    # 按 MRI 清单过滤
    if os.path.exists(mri_list):
        df = filter_by_mri_list(df, mri_list)
    else:
        print(f"  [警告] MRI 清单未找到 ({mri_list})，保留全部目标访视")

    # 只保留目标访视
    target = {"bl", "m12", "m24", "m36", "m48", "m60"}
    if "VISCODE" in df.columns:
        df = df[df["VISCODE"].isin(target)].copy()

    # 去重：同一 PTID+VISCODE 按描述优先级选最优一条
    key = [c for c in ["PTID", "RID", "VISCODE"] if c in df.columns]
    if key:
        n = len(df)
        sort_cols = key + (["_PRI", "IMAGEUID"] if "_PRI" in df.columns else ["IMAGEUID"])
        df = (df.sort_values(sort_cols)
                .drop_duplicates(subset=[c for c in key if c != "IMAGEUID"],
                                 keep="first")
                .reset_index(drop=True))
        if "_PRI" in df.columns:
            df = df.drop(columns=["_PRI"])
        if len(df) < n:
            print(f"  去重（按描述优先级）: {n} → {len(df)}")

    if df.empty:
        print("[失败] 过滤后无数据")
        return

    print(f"\n目标访视 FDG-PET: {len(df)} 条")
    if "VISCODE" in df.columns:
        print(df["VISCODE"].value_counts().sort_index().to_string())

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n保存至: {output_path}")

    id_list = df["IMAGEUID"].dropna().astype(int).tolist()
    with open(output_path.replace(".csv", "_ids_only.txt"), "w") as f:
        f.write("\n".join(map(str, id_list)))
    with open(output_path.replace(".csv", "_ids_comma.txt"), "w") as f:
        f.write(",".join(map(str, id_list)))

    print(f"共 {len(id_list)} 个 FDG-PET Image ID")
    print(f"逗号分隔（粘贴用）: {output_path.replace('.csv','_ids_comma.txt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rda_dir",  required=True)
    parser.add_argument("--mri_list", default="data/mri_download_list.csv")
    parser.add_argument("--output",   default="data/pet_download_list.csv")
    args = parser.parse_args()
    generate_pet_download_list(args.rda_dir, args.mri_list, args.output)
