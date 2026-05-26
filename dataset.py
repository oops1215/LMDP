"""
PyTorch Dataset for LMDP-Net

__getitem__ 返回的 sample 包含：
  mri_seq      : list of (1,128,160,128) np.float32 or None（每访视）
  pet_seq      : list of (1,128,160,128) np.float32 or None
  non_img_seq  : (T, NON_IMG_DIM) np.float32  [bio(6) + demo(4) + gen(3)]
  bio_mask_seq : (T, 6) np.float32    生物标志物观测掩码
  mod_avail    : (T, 2) np.float32    [mri_avail, pet_avail]
  delta_seq    : (T, 1) np.float32    距上次访视月数
  dx_seq       : (T,)   np.int64      诊断标签
  length       : int                  实际访视数

collate_fn 将 batch 打包为：
  mri_seq      : (B, T_max, 1, 128, 160, 128) Tensor
  pet_seq      : (B, T_max, 1, 128, 160, 128) Tensor
  non_img_seq  : (B, T_max, NON_IMG_DIM) Tensor
  ...（其余字段填充至 T_max）
  lengths      : (B,) LongTensor
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ADNIDataset(Dataset):
    """
    参数
    ----
    subject_list   : 受试者 PTID 列表
    subject_data   : preprocess_tabular 生成的数据字典
    bio_scaler     : 已 fit 的 StandardScaler（只对生物标志物）
    age_scaler     : 已 fit 的 StandardScaler（对年龄）
    edu_scaler     : 已 fit 的 StandardScaler（对教育年限）
    load_images    : False 时跳过图像加载（快速调试非图像模型）
    max_seq_len    : 最大序列长度（超过则截断）
    """

    def __init__(self,
                 subject_list:  List[str],
                 subject_data:  Dict,
                 bio_scaler:    StandardScaler,
                 age_scaler:    StandardScaler,
                 edu_scaler:    StandardScaler,
                 load_images:   bool = True,
                 max_seq_len:   int  = Config.MAX_SEQ_LEN):
        self.subjects    = subject_list
        self.data        = subject_data
        self.bio_scaler  = bio_scaler
        self.age_scaler  = age_scaler
        self.edu_scaler  = edu_scaler
        self.load_images = load_images
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict:
        ptid    = self.subjects[idx]
        subject = self.data[ptid]
        visits  = subject["visits"][:self.max_seq_len]
        T       = len(visits)

        mri_list, pet_list = [], []
        non_img_list, bio_mask_list = [], []
        mod_avail_list, delta_list, dx_list = [], [], []

        prev_month = 0
        for i, v in enumerate(visits):
            # ── 时间间隔 ─────────────────────────────────────────────────
            delta = v["month"] - prev_month if i > 0 else 0
            prev_month = v["month"]
            delta_list.append([float(delta)])

            # ── 诊断 ─────────────────────────────────────────────────────
            dx_list.append(v["dx"])

            # ── 生物标志物（element-wise z-score）────────────────────────
            bio_raw  = v["bio_raw"].copy()     # (6,)
            bio_mask = v["bio_mask"].copy()    # (6,)
            obs_idx  = bio_mask > 0
            # 直接用 scaler 的 mean_/var_ 做逐元素标准化，
            # 避免缺失维度导致形状不匹配
            if obs_idx.any() and self.bio_scaler is not None:
                mu  = self.bio_scaler.mean_[obs_idx]
                std = np.sqrt(self.bio_scaler.var_[obs_idx] + 1e-8)
                bio_raw[obs_idx] = (bio_raw[obs_idx] - mu) / std
            bio_mask_list.append(bio_mask)

            # ── 人口统计学（z-score）──────────────────────────────────────
            age_z = float(self.age_scaler.transform([[v["demo_raw"][0]]])[0, 0]) \
                if self.age_scaler else v["demo_raw"][0]
            edu_z = float(self.edu_scaler.transform([[v["demo_raw"][1]]])[0, 0]) \
                if self.edu_scaler else v["demo_raw"][1]

            demo_vec = np.array([age_z, edu_z], dtype=np.float32)
            gender   = v["gender_oh"]    # (2,)
            apoe4    = v["apoe4_oh"]     # (3,)

            non_img = np.concatenate([bio_raw, demo_vec, gender, apoe4]).astype(np.float32)
            non_img_list.append(non_img)

            # ── 图像模态可用性 ────────────────────────────────────────────
            mri_avail = float(v["mri_path"] is not None)
            pet_avail = float(v["pet_path"] is not None)
            mod_avail_list.append([mri_avail, pet_avail])

            # ── 图像数据 ──────────────────────────────────────────────────
            if self.load_images:
                mri = self._load_image(v["mri_path"])
                pet = self._load_image(v["pet_path"])
            else:
                mri = pet = None
            mri_list.append(mri)
            pet_list.append(pet)

        return {
            "ptid"        : ptid,
            "mri_list"    : mri_list,           # list of T arrays (1,128,160,128) or None
            "pet_list"    : pet_list,
            "non_img_seq" : np.stack(non_img_list),          # (T, 13)
            "bio_mask_seq": np.stack(bio_mask_list),          # (T, 6)
            "mod_avail"   : np.array(mod_avail_list, dtype=np.float32),  # (T, 2)
            "delta_seq"   : np.array(delta_list, dtype=np.float32),      # (T, 1)
            "dx_seq"      : np.array(dx_list, dtype=np.int64),            # (T,)
            "length"      : T,
        }

    @staticmethod
    def _load_image(path: Optional[str]) -> Optional[np.ndarray]:
        """加载预处理后的 .npy 图像，返回 (1, 128, 160, 128) float32 数组。"""
        if path is None or not os.path.exists(path):
            return None
        img = np.load(path).astype(np.float32)
        if img.ndim == 3:
            img = img[np.newaxis]   # (1, D, H, W)
        return img


# ─── Collate 函数（变长序列打包）────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    将不同长度的序列填充到同一长度，堆叠成 Tensor batch。
    图像序列填充为全零张量。
    """
    B         = len(batch)
    T_max     = max(s["length"] for s in batch)
    D, H, W   = Config.TARGET_SHAPE   # 128, 160, 128

    non_img_dim  = Config.NON_IMG_DIM
    bio_dim      = Config.BIOMARKER_DIM

    # 预分配
    mri_batch  = torch.zeros(B, T_max, 1, D, H, W)
    pet_batch  = torch.zeros(B, T_max, 1, D, H, W)
    non_img    = torch.zeros(B, T_max, non_img_dim)
    bio_mask   = torch.zeros(B, T_max, bio_dim)
    mod_avail  = torch.zeros(B, T_max, 2)
    delta_seq  = torch.zeros(B, T_max, 1)
    dx_seq     = torch.full((B, T_max), -1, dtype=torch.long)
    lengths    = torch.tensor([s["length"] for s in batch], dtype=torch.long)
    has_images = any(s["mri_list"][0] is not None or s["pet_list"][0] is not None
                     for s in batch)

    for b, sample in enumerate(batch):
        T = sample["length"]
        non_img[b, :T]   = torch.from_numpy(sample["non_img_seq"])
        bio_mask[b, :T]  = torch.from_numpy(sample["bio_mask_seq"])
        mod_avail[b, :T] = torch.from_numpy(sample["mod_avail"])
        delta_seq[b, :T] = torch.from_numpy(sample["delta_seq"])
        dx_seq[b, :T]    = torch.from_numpy(sample["dx_seq"])

        for t, (mri, pet) in enumerate(zip(sample["mri_list"], sample["pet_list"])):
            if mri is not None:
                mri_batch[b, t] = torch.from_numpy(mri)
            if pet is not None:
                pet_batch[b, t] = torch.from_numpy(pet)

    result = {
        "non_img_seq"  : non_img,
        "bio_mask_seq" : bio_mask,
        "mod_avail"    : mod_avail,
        "delta_seq"    : delta_seq,
        "dx_seq"       : dx_seq,
        "lengths"      : lengths,
    }
    if has_images:
        result["mri_seq"] = mri_batch
        result["pet_seq"] = pet_batch

    return result


# ─── 构建 DataLoader 的工厂函数 ──────────────────────────────────────────────

def build_dataloaders(processed_data_path: str = Config.TAB_PROCESSED_PATH,
                      k_fold: int = Config.K_FOLDS,
                      fold_idx: int = 0,
                      batch_size: int = Config.BATCH_SIZE,
                      load_images: bool = True,
                      num_workers: int = 4,
                      seed: int = Config.SEED) -> Tuple[DataLoader, DataLoader]:
    """
    构建 K-fold 的 train/val DataLoader。

    返回
    ----
    train_loader, val_loader
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    # 加载预处理后的数据
    with open(processed_data_path, "rb") as f:
        packed = pickle.load(f)
    subject_data = packed["subjects"]
    bio_scaler   = packed["bio_scaler"]
    age_scaler   = packed["age_scaler"]
    edu_scaler   = packed["edu_scaler"]

    all_ptids = sorted(subject_data.keys())
    np.random.shuffle(all_ptids)

    # 5-fold split
    fold_size = len(all_ptids) // k_fold
    val_start = fold_idx * fold_size
    val_end   = val_start + fold_size if fold_idx < k_fold - 1 else len(all_ptids)

    val_ptids   = all_ptids[val_start:val_end]
    train_ptids = all_ptids[:val_start] + all_ptids[val_end:]

    # 在训练集上重新 fit scaler（避免 data leakage）
    train_data   = {p: subject_data[p] for p in train_ptids}
    train_scaler = _refit_scalers(train_data)

    train_ds = ADNIDataset(train_ptids, subject_data,
                            train_scaler["bio"], train_scaler["age"], train_scaler["edu"],
                            load_images=load_images)
    val_ds   = ADNIDataset(val_ptids,   subject_data,
                            train_scaler["bio"], train_scaler["age"], train_scaler["edu"],
                            load_images=load_images)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, collate_fn=collate_fn,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                               shuffle=False, collate_fn=collate_fn,
                               num_workers=num_workers, pin_memory=True)

    print(f"Fold {fold_idx}: train={len(train_ptids)}, val={len(val_ptids)}")
    return train_loader, val_loader


def _refit_scalers(subject_data: Dict) -> Dict:
    """在给定受试者子集上重新拟合 StandardScaler。"""
    # 收集完整的 6D 生物标志物行（全部 6 个均可观测），用于 6D scaler 拟合
    bio_rows, age_vals, edu_vals = [], [], []
    for subj in subject_data.values():
        for v in subj["visits"]:
            obs_mask = v["bio_mask"] > 0
            if obs_mask.all():   # 只用 6 维全齐的行，保持 scaler 维度为 6
                bio_rows.append(v["bio_raw"])
            age_vals.append([v["demo_raw"][0]])
            edu_vals.append([v["demo_raw"][1]])

    bio_scaler = StandardScaler()
    bio_scaler.fit(np.array(bio_rows) if bio_rows else np.zeros((1, 6)))

    age_scaler = StandardScaler()
    age_scaler.fit(np.array(age_vals) if age_vals else [[0]])

    edu_scaler = StandardScaler()
    edu_scaler.fit(np.array(edu_vals) if edu_vals else [[0]])

    return {"bio": bio_scaler, "age": age_scaler, "edu": edu_scaler}
