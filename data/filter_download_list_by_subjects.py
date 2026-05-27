"""
用 tabular_processed.pkl 中的有效受试者名单过滤 MRI/PET 下载清单，
只保留最终会被模型用到的受试者，减少不必要的下载量。

用法：
  python data/filter_download_list_by_subjects.py \\
      --tabular  data/tabular_processed.pkl \\
      --mri_list data/mri_download_list.csv \\
      --pet_list data/pet_download_list.csv
"""

import os
import sys
import pickle
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def filter_list(input_csv: str, valid_ptids: set, label: str):
    if not os.path.exists(input_csv):
        print(f"  [跳过] {input_csv} 不存在")
        return

    df = pd.read_csv(input_csv)
    n_before = len(df)
    n_subj_before = df["PTID"].nunique() if "PTID" in df.columns else "?"

    df_filtered = df[df["PTID"].astype(str).isin(valid_ptids)].copy()
    n_after = len(df_filtered)
    n_subj_after = df_filtered["PTID"].nunique() if "PTID" in df_filtered.columns else "?"

    # 覆盖原文件
    df_filtered.to_csv(input_csv, index=False)

    # 同步更新 ids 文件
    if "IMAGEUID" in df_filtered.columns:
        id_list = df_filtered["IMAGEUID"].dropna().astype(int).tolist()
        base = input_csv.replace(".csv", "")
        with open(f"{base}_ids_only.txt", "w") as f:
            f.write("\n".join(map(str, id_list)))
        with open(f"{base}_ids_comma.txt", "w") as f:
            f.write(",".join(map(str, id_list)))
        print(f"  {label}: {n_subj_before} 人 {n_before} 条 → {n_subj_after} 人 {n_after} 条"
              f"（{len(id_list)} 个 Image ID）")
    else:
        print(f"  {label}: {n_before} 条 → {n_after} 条")


def main(tabular_path: str, mri_list: str, pet_list: str):
    print(f"读取 {tabular_path} ...")
    with open(tabular_path, "rb") as f:
        packed = pickle.load(f)
    subject_data = packed["subjects"]
    valid_ptids = set(subject_data.keys())
    print(f"  有效受试者: {len(valid_ptids)} 人")

    print("\n过滤下载清单 ...")
    filter_list(mri_list, valid_ptids, "MRI")
    filter_list(pet_list, valid_ptids, "PET")

    print("\n完成。用过滤后的 _ids_comma.txt 去 ADNI IDA 下载即可。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabular",  default="data/tabular_processed.pkl")
    parser.add_argument("--mri_list", default="data/mri_download_list.csv")
    parser.add_argument("--pet_list", default="data/pet_download_list.csv")
    args = parser.parse_args()
    main(args.tabular, args.mri_list, args.pet_list)
