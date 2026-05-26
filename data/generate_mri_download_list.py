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
    从所有 UCSFFSX*.rda 文件提取 RID, VISCODE, IMAGEUID（及可选的 EXAMDATE、FLDSTRENG）。
    这些 IMAGEUID 就是 ADNI IDA 中的 Image Data ID。
    """
    fs_files = [
        "UCSFFSX.rda",
        "UCSFFSX51ALL.rda",
        "UCSFFSX51_ADNI1_3T.rda",
        "UCSFFSX6.rda",
        "UCSFFSX7.rda",
    ]
    frames = []
    for fname in fs_files:
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
            sub["EXAMDATE"]  = pd.to_datetime(df["EXAMDATE"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "FLDSTRENG" in df.columns:
            sub["FLDSTRENG"] = df["FLDSTRENG"]   # 磁场强度（1.5 / 3T）
        if "PTID" in df.columns:
            sub["PTID"] = df["PTID"]

        # QC 过滤
        qc = next((c for c in ["OVERALLQC","LHIPQC","HIPPOQC"] if c in df.columns), None)
        if qc:
            sub = sub[df[qc].astype(str).str.strip().str.upper().isin(
                ["PASS","1","1.0","TRUE"])]

        sub["SOURCE"] = fname
        frames.append(sub)
        print(f"  {fname}: {len(sub)} 条有效 IMAGEUID")

    if not frames:
        print("[警告] 未从任何 UCSFFSX 文件中提取到 IMAGEUID")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # 同一 RID+VISCODE 保留最新处理结果（最后加载的文件优先）
    combined = combined.drop_duplicates(subset=["RID","VISCODE"], keep="last")
    combined = combined.sort_values(["RID","VISCODE"]).reset_index(drop=True)
    print(f"\n汇总: {len(combined)} 条记录，{combined['RID'].nunique()} 名受试者")
    return combined


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
    if "FLDSTRENG" in df_target.columns:
        print(f"\n磁场强度分布:")
        print(df_target["FLDSTRENG"].value_counts().to_string())

    # 生成 ADNI IDA 搜索用的 Image ID 列表（逗号分隔）
    id_list = df_target["IMAGEUID"].dropna().astype(int).tolist()
    id_file = output_path.replace(".csv", "_ids_only.txt")
    with open(id_file, "w") as f:
        f.write("\n".join(str(i) for i in id_list))
    print(f"\n纯 ID 列表（粘贴到 ADNI IDA 搜索框）: {id_file}")
    print(f"共 {len(id_list)} 个 Image ID")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rda_dir", required=True,
                        help="ADNIMERGE2 的 data/ 目录")
    parser.add_argument("--output",  default="data/mri_download_list.csv")
    args = parser.parse_args()
    generate_download_list(args.rda_dir, args.output)
