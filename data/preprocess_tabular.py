"""
对 ADNIMERGE.csv 进行完整预处理，生成模型可直接使用的 tabular_processed.pkl。

产生的数据结构（dict，键为 PTID）：
  {
    "002_S_0413": {
      "visits": [
        {
          "viscode"    : "bl",
          "month"      : 0,
          "exam_date"  : "2005-09-08",
          "dx"         : 1,          # 0=CN, 1=MCI, 2=AD
          "biomarkers" : np.ndarray(6,),  # z-score 后的 [Vent, Hipp, WB, Ent, Fus, MidT]
          "demographics": np.ndarray(4,), # [age_z, edu_z, gender_m, gender_f]
          "genetics"   : np.ndarray(3,),  # APOE4 one-hot
          "bio_mask"   : np.ndarray(6,),  # 1=observed, 0=missing
          "mri_path"   : "data/mri_preprocessed/002_S_0413_bl.npy" or None,
          "pet_path"   : "data/pet_preprocessed/002_S_0413_bl.npy" or None,
        },
        ...
      ]
    },
    ...
  }

论文设置（Section IV.A）：
  - 保留 M0, M12, M24, M36, M48, M60 六个时间点
  - 最少有 2 次访视
  - 排除诊断"逆转"的受试者（MCI→CN 或 AD→MCI/CN）
  - 最终 1369 人，5768 次访视（stable 971，progressive 398）
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# ─── 诊断编码 ─────────────────────────────────────────────────────────────────

def encode_dx(dx_str: str) -> int:
    if isinstance(dx_str, float):   # NaN
        return -1
    dx_str = str(dx_str).strip()
    return Config.DX_MAP.get(dx_str, -1)


# ─── 判断诊断序列是否单调（非逆转）────────────────────────────────────────────

def is_monotone(dx_seq: list) -> bool:
    """返回 True 表示诊断不逆转（允许 CN→CN→MCI→AD）。"""
    valid = [d for d in dx_seq if d >= 0]
    return all(valid[i] <= valid[i + 1] for i in range(len(valid) - 1))


# ─── 主预处理函数 ──────────────────────────────────────────────────────────────

def preprocess_tabular(adnimerge_path: str = Config.ADNIMERGE_PATH,
                       mri_prep_dir: str = Config.MRI_PREP_DIR,
                       pet_prep_dir: str = Config.PET_PREP_DIR,
                       output_path: str = Config.TAB_PROCESSED_PATH) -> dict:
    """
    读取 ADNIMERGE.csv，过滤、编码、归一化，生成模型所需的数据字典。
    """
    print(f"读取 {adnimerge_path} ...")
    df = pd.read_csv(adnimerge_path, low_memory=False)
    print(f"  原始记录数: {len(df)}")

    # ── 1. 只保留目标访视 ────────────────────────────────────────────────────
    df = df[df["VISCODE"].isin(Config.VISIT_CODES)].copy()
    print(f"  过滤到目标访视后: {len(df)} 条")

    # ── 2. 编码诊断 ───────────────────────────────────────────────────────────
    df["DX_int"] = df["DX"].apply(encode_dx)

    # 某些行 DX 为空，但 DX_bl 不为空（基线诊断），用基线诊断填充
    if "DX_bl" in df.columns:
        mask_missing = df["DX_int"] == -1
        df.loc[mask_missing, "DX_int"] = df.loc[mask_missing, "DX_bl"].apply(encode_dx)

    # ── 3. 去除 DX 完全未知的受试者 ─────────────────────────────────────────
    df = df[df["DX_int"] >= 0].copy()

    # ── 4. 对每个受试者按月份排序，过滤逆转 ─────────────────────────────────
    df["month"] = df["VISCODE"].map(Config.VISIT_MONTHS)
    df = df.sort_values(["PTID", "month"])

    valid_ptids = []
    for ptid, grp in df.groupby("PTID"):
        dx_seq = grp["DX_int"].tolist()
        if len(grp) < 2:
            continue
        if not is_monotone(dx_seq):
            continue
        valid_ptids.append(ptid)

    df = df[df["PTID"].isin(valid_ptids)].copy()
    print(f"  过滤逆转/不足2次访视后: {df['PTID'].nunique()} 人，{len(df)} 条")

    # ── 5. 年龄：只有基线年龄，后续按时间间隔递增 ───────────────────────────
    baseline_age = df[df["VISCODE"] == "bl"][["PTID", "AGE"]].set_index("PTID")["AGE"]
    def compute_age(row):
        base = baseline_age.get(row["PTID"], np.nan)
        if np.isnan(base):
            return np.nan
        return base + row["month"] / 12.0
    df["AGE_visit"] = df.apply(compute_age, axis=1)

    # ── 6. 生物标志物：除以颅内容积（ICV）归一化 ────────────────────────────
    for col in Config.BIOMARKER_COLS:
        if col in df.columns and Config.ICV_COL in df.columns:
            df[f"{col}_icv"] = df[col] / df[Config.ICV_COL]
        elif col in df.columns:
            df[f"{col}_icv"] = df[col]
    biomarker_icv_cols = [f"{c}_icv" for c in Config.BIOMARKER_COLS]

    # ── 7. Z-score 归一化（在所有训练数据上拟合，这里先全局拟合） ───────────
    #    注意：在正式 K-fold 训练时，scaler 应只在 train fold 上 fit。
    #    这里保存原始 ICV-normalized 值；实际 train.py 中再做 z-score。
    #    但我们先把全局统计量存起来，方便验证。

    bio_arr = df[biomarker_icv_cols].values.astype(np.float32)
    age_arr = df["AGE_visit"].values.astype(np.float32).reshape(-1, 1)
    edu_arr = df["PTEDUCAT"].values.astype(np.float32).reshape(-1, 1) \
        if "PTEDUCAT" in df.columns else np.zeros((len(df), 1), dtype=np.float32)

    # 全局 scaler（仅用于 sanity check，训练时重新在 fold 上 fit）
    bio_scaler = StandardScaler()
    bio_scaler.fit(bio_arr[~np.isnan(bio_arr).any(axis=1)])
    age_scaler = StandardScaler()
    age_scaler.fit(age_arr[~np.isnan(age_arr).any(axis=0)])
    edu_scaler = StandardScaler()
    edu_scaler.fit(edu_arr[~np.isnan(edu_arr).any(axis=0)])

    # ── 8. 性别 one-hot ──────────────────────────────────────────────────────
    def gender_oh(g):
        if str(g).lower() in ("male", "m", "1"):
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    # ── 9. APOE4 one-hot ────────────────────────────────────────────────────
    def apoe4_oh(a):
        try:
            a = int(a)
        except (ValueError, TypeError):
            a = 0
        oh = np.zeros(3, dtype=np.float32)
        oh[min(a, 2)] = 1.0
        return oh

    # ── 10. 构建受试者数据字典 ───────────────────────────────────────────────
    print("构建受试者数据字典 ...")
    subject_data = {}

    # 按 PTID 分组
    for ptid, grp in tqdm(df.groupby("PTID"), desc="Processing subjects"):
        grp = grp.sort_values("month").reset_index(drop=True)
        visits = []

        gender_vec = gender_oh(grp["PTGENDER"].iloc[0] if "PTGENDER" in grp.columns else "Female")

        for _, row in grp.iterrows():
            viscode = row["VISCODE"]
            month   = int(row["month"])

            # ── 生物标志物（ICV 归一化后的原始值，训练时再 z-score）──────────
            bio_raw = np.array(
                [row.get(f"{c}_icv", np.nan) for c in Config.BIOMARKER_COLS],
                dtype=np.float32
            )
            bio_mask = (~np.isnan(bio_raw)).astype(np.float32)
            bio_raw  = np.where(np.isnan(bio_raw), 0.0, bio_raw)

            # ── 人口统计学（原始值，训练时 z-score）──────────────────────────
            age_val = float(row["AGE_visit"]) if not np.isnan(row["AGE_visit"]) else 0.0
            edu_val = float(row["PTEDUCAT"]) if "PTEDUCAT" in row and not pd.isna(row["PTEDUCAT"]) else 0.0
            demo_raw = np.array([age_val, edu_val], dtype=np.float32)  # 未归一化

            # ── 遗传学 ───────────────────────────────────────────────────────
            apoe4 = apoe4_oh(row.get("APOE4", 0))

            # ── 图像路径 ─────────────────────────────────────────────────────
            key = f"{ptid}_{viscode}"
            mri_path = os.path.join(mri_prep_dir, f"{key}.npy")
            pet_path = os.path.join(pet_prep_dir, f"{key}.npy")
            mri_path = mri_path if os.path.exists(mri_path) else None
            pet_path = pet_path if os.path.exists(pet_path) else None

            visits.append({
                "viscode"     : viscode,
                "month"       : month,
                "exam_date"   : str(row.get("EXAMDATE", "")),
                "dx"          : int(row["DX_int"]),
                "bio_raw"     : bio_raw,      # shape (6,), ICV-normalized, pre-z-score
                "bio_mask"    : bio_mask,     # shape (6,), 1=observed
                "demo_raw"    : demo_raw,     # shape (2,), [age, edu], pre-z-score
                "gender_oh"   : gender_vec,   # shape (2,)
                "apoe4_oh"    : apoe4,        # shape (3,)
                "mri_path"    : mri_path,
                "pet_path"    : pet_path,
            })

        subject_data[ptid] = {
            "visits": visits,
            "n_visits": len(visits),
            "baseline_dx": visits[0]["dx"],
            "final_dx": visits[-1]["dx"],
            "progressive": int(visits[0]["dx"] != visits[-1]["dx"]),
        }

    n_total     = len(subject_data)
    n_prog      = sum(v["progressive"] for v in subject_data.values())
    n_stable    = n_total - n_prog
    total_visits = sum(v["n_visits"] for v in subject_data.values())
    print(f"\n=== 数据集统计 ===")
    print(f"  受试者总数  : {n_total}  (stable={n_stable}, progressive={n_prog})")
    print(f"  总访视数    : {total_visits}")

    # ── 11. 保存 ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "subjects": subject_data,
            "bio_scaler": bio_scaler,
            "age_scaler": age_scaler,
            "edu_scaler": edu_scaler,
        }, f)
    print(f"\n保存至 {output_path}")
    return subject_data


if __name__ == "__main__":
    preprocess_tabular()
