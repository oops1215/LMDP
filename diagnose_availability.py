"""
诊断脚本：检查数据集图像可用率 + 训练中 encoder logvar 情况
用法：python diagnose_availability.py
"""
import sys, pickle
import torch
import numpy as np
sys.path.insert(0, '.')
from config import Config

# ── 1. 数据层：图像路径可用率 ──────────────────────────────────────────────
print("=" * 60)
print("1. 数据集图像可用率（path 层面）")
print("=" * 60)

with open(Config.TAB_PROCESSED_PATH, 'rb') as f:
    data = pickle.load(f)

# 自动探测结构
print(f"pickle 顶层类型: {type(data)}")
if isinstance(data, dict):
    print(f"顶层 keys: {list(data.keys())[:10]}")
    # 尝试找到受试者字典：值为 dict 且含 'visits'
    root = data
    sample_key = next(iter(root))
    sample_val = root[sample_key]
    if isinstance(sample_val, dict) and 'visits' in sample_val:
        subject_dict = root                     # 顶层直接是 {ptid: {...}}
    elif 'data' in root:
        subject_dict = root['data']
    elif 'subjects' in root:
        subject_dict = root['subjects']
    else:
        # 找第一个值是 dict-with-visits 的 key
        subject_dict = None
        for k, v in root.items():
            if isinstance(v, dict) and any(
                isinstance(vv, dict) and 'visits' in vv
                for vv in (v.values() if isinstance(v, dict) else [])
            ):
                subject_dict = v
                print(f"  → 受试者数据在 key='{k}'")
                break
        if subject_dict is None:
            print("无法自动定位受试者字典，请检查 pickle 结构")
            raise SystemExit(1)
else:
    print(f"顶层不是 dict，而是 {type(data)}，请检查 pickle 结构")
    raise SystemExit(1)

subjects = list(subject_dict.keys())
total_visits = mri_n = pet_n = both_n = neither_n = 0

for ptid in subjects:
    for v in subject_dict[ptid]['visits']:
        total_visits += 1
        has_mri = v.get('mri_path') is not None
        has_pet = v.get('pet_path') is not None
        mri_n     += int(has_mri)
        pet_n     += int(has_pet)
        both_n    += int(has_mri and has_pet)
        neither_n += int(not has_mri and not has_pet)

print(f"总受试者: {len(subjects)}")
print(f"总访次:   {total_visits}")
print(f"有 MRI:   {mri_n:4d} / {total_visits} = {100*mri_n/total_visits:.1f}%")
print(f"有 PET:   {pet_n:4d} / {total_visits} = {100*pet_n/total_visits:.1f}%")
print(f"MRI+PET:  {both_n:4d} / {total_visits} = {100*both_n/total_visits:.1f}%")
print(f"无图像:   {neither_n:4d} / {total_visits} = {100*neither_n/total_visits:.1f}%")

# ── 2. 模型层：encoder logvar 情况（加载一个 checkpoint）──────────────────
print()
print("=" * 60)
print("2. Encoder logvar 诊断（需要有已训练的 checkpoint）")
print("=" * 60)

import glob, os
ckpts = sorted(glob.glob("checkpoints/**/*.pt", recursive=True) +
               glob.glob("*.pt") + glob.glob("checkpoints/*.pt"))
if not ckpts:
    print("未找到 checkpoint，跳过 logvar 诊断")
    print("提示：若 prior≈98% 纯因图像缺失，属正常；")
    print("      若有图像但 prior 仍高，则是 encoder 输出 logvar 过大（posterior collapse）")
else:
    from models.lmdp_net import LMDPNet
    from dataset import build_dataloaders

    ckpt_path = ckpts[-1]
    print(f"加载: {ckpt_path}")
    state = torch.load(ckpt_path, map_location='cpu')

    model = LMDPNet(
        latent_dim=Config.LATENT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        non_img_dim=Config.NON_IMG_DIM,
        num_classes=Config.NUM_CLASSES,
        biomarker_dim=Config.BIOMARKER_DIM,
    )
    model.load_state_dict(state.get('model_state_dict', state))
    model.eval()

    # 用随机噪声图像探测 logvar
    dummy_mri = torch.randn(4, 1, 96, 114, 96)
    dummy_pet = torch.randn(4, 1, 96, 114, 96)
    with torch.no_grad():
        _, mri_lv = model.m3vae.mri_encoder(dummy_mri)
        _, pet_lv = model.m3vae.pet_encoder(dummy_pet)
    print(f"MRI encoder logvar: mean={mri_lv.mean():.2f}, std={mri_lv.std():.2f}")
    print(f"PET encoder logvar: mean={pet_lv.mean():.2f}, std={pet_lv.std():.2f}")
    print(f"  → exp(-logvar) MRI: {torch.exp(-mri_lv).mean():.4f}  (>>1 = 高置信, <<1 = 崩塌)")
    print(f"  → exp(-logvar) PET: {torch.exp(-pet_lv).mean():.4f}")

print()
print("结论判断：")
print("  · 若'无图像'≈98% → prior 高是数据决定的，属正常")
print("  · 若'无图像'<<98% 但 prior 仍 98% → encoder logvar 过大，需增大 kl_weight")
