"""
数据预处理验证脚本

检查项：
  1. tabular_processed.pkl 结构完整性
  2. 生物标志物数值范围、NaN 率
  3. 诊断标签分布 & 单调性
  4. 时间间隔合理性
  5. 图像文件存在性 & 数值范围
  6. Dataset / collate_fn 输出 shape & 数值
  7. 影像模态可用率

用法：
  python check_data.py                      # 全部检查（不加载图像内容）
  python check_data.py --load_images        # 同时采样几张图像做数值检查（慢）
  python check_data.py --n_img_samples 20   # 随机抽 20 张图像验证（默认 5）
"""

import os
import sys
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def check(cond, msg_pass, msg_fail):
    if cond:
        print(f"  {PASS} {msg_pass}")
    else:
        print(f"  {FAIL} {msg_fail}")
    return cond


def warn_if(cond, msg):
    if cond:
        print(f"  {WARN} {msg}")


# ── 1. PKL 结构检查 ─────────────────────────────────────────────────────────

def check_pkl(packed):
    print("\n=== 1. PKL 结构检查 ===")
    check("subjects"   in packed, "subjects 键存在",   "缺少 subjects 键")
    check("bio_scaler" in packed, "bio_scaler 键存在", "缺少 bio_scaler 键")
    check("age_scaler" in packed, "age_scaler 键存在", "缺少 age_scaler 键")
    check("edu_scaler" in packed, "edu_scaler 键存在", "缺少 edu_scaler 键")

    subjects = packed["subjects"]
    n = len(subjects)
    print(f"  {INFO} 受试者总数: {n}")
    check(n >= 100, f"受试者数量合理 ({n})", f"受试者数量过少 ({n})，预期 ≥ 100")

    # 检查第一个受试者的字段
    ptid0 = next(iter(subjects))
    s0    = subjects[ptid0]
    check("visits"    in s0, "visits 字段存在",    "缺少 visits 字段")
    check("n_visits"  in s0, "n_visits 字段存在",  "缺少 n_visits 字段")
    check("progressive" in s0, "progressive 字段存在", "缺少 progressive 字段")

    v0 = s0["visits"][0]
    required_keys = ["viscode", "month", "dx", "bio_raw", "bio_mask",
                     "demo_raw", "gender_oh", "apoe4_oh", "mri_path", "pet_path"]
    for k in required_keys:
        check(k in v0, f"visit 含 '{k}' 字段", f"visit 缺少 '{k}' 字段")

    return subjects


# ── 2. 生物标志物统计 ────────────────────────────────────────────────────────

def check_biomarkers(subjects):
    print("\n=== 2. 生物标志物数值检查 ===")
    bio_names = Config.BIOMARKER_COLS
    all_bio   = []
    all_masks = []
    for s in subjects.values():
        for v in s["visits"]:
            all_bio.append(v["bio_raw"])
            all_masks.append(v["bio_mask"])

    bio_arr  = np.array(all_bio)    # (N, 6)
    mask_arr = np.array(all_masks)  # (N, 6)

    print(f"  {INFO} 总访视数: {len(bio_arr)}")
    print(f"  {'特征':<20} {'均值':>10} {'标准差':>10} {'最小':>10} {'最大':>10} {'覆盖率':>8}")
    print("  " + "-"*68)
    for i, name in enumerate(bio_names):
        obs    = mask_arr[:, i] > 0
        vals   = bio_arr[obs, i]
        cov    = obs.mean() * 100
        if len(vals) == 0:
            print(f"  {name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {cov:>7.1f}%")
            continue
        mu, sd = vals.mean(), vals.std()
        vmin, vmax = vals.min(), vals.max()
        print(f"  {name:<20} {mu:>10.4f} {sd:>10.4f} {vmin:>10.4f} {vmax:>10.4f} {cov:>7.1f}%")
        warn_if(vmin < 0, f"{name} 含负值（ICV 归一化后不应为负），请检查预处理")
        warn_if(sd < 1e-6, f"{name} 标准差接近 0，可能数据未正确加载")
        warn_if(cov < 10, f"{name} 覆盖率低于 10%，该特征几乎全部缺失")

    # 检查 bio_raw 中缺失位置是否已清零
    missing_nonzero = ((mask_arr == 0) & (bio_arr != 0)).sum()
    check(missing_nonzero == 0,
          "缺失位置已清零",
          f"缺失位置有 {missing_nonzero} 个非零值（应为 0，会干扰插补模块）")


# ── 3. 诊断标签分布 ─────────────────────────────────────────────────────────

def check_dx(subjects):
    print("\n=== 3. 诊断标签分布 ===")
    dx_counts = {0: 0, 1: 0, 2: 0, -1: 0}
    monotone_ok = 0
    monotone_bad = 0
    visit_counts = []

    for s in subjects.values():
        visits = s["visits"]
        visit_counts.append(len(visits))
        dx_seq = [v["dx"] for v in visits]
        for d in dx_seq:
            dx_counts[d] = dx_counts.get(d, 0) + 1

        valid = [d for d in dx_seq if d >= 0]
        mono  = all(valid[i] <= valid[i+1] for i in range(len(valid)-1))
        if mono:
            monotone_ok  += 1
        else:
            monotone_bad += 1

    total_valid = dx_counts[0] + dx_counts[1] + dx_counts[2]
    print(f"  {INFO} 诊断分布: CN={dx_counts[0]}({100*dx_counts[0]/max(total_valid,1):.1f}%) "
          f"MCI={dx_counts[1]}({100*dx_counts[1]/max(total_valid,1):.1f}%) "
          f"AD={dx_counts[2]}({100*dx_counts[2]/max(total_valid,1):.1f}%)")
    warn_if(dx_counts[-1] > 0, f"仍有 {dx_counts[-1]} 个访视的 DX=-1（未知），会被损失函数跳过")

    check(monotone_bad == 0,
          f"所有受试者诊断序列单调 ({monotone_ok} 人)",
          f"{monotone_bad} 个受试者诊断序列存在逆转（已过滤但仍在数据中？）")

    vc = np.array(visit_counts)
    print(f"  {INFO} 访视次数: 均值={vc.mean():.2f}  中位={np.median(vc):.0f}  "
          f"min={vc.min()}  max={vc.max()}")
    check(vc.min() >= 2, f"所有受试者 ≥ 2 次访视", f"{(vc < 2).sum()} 个受试者只有 1 次访视")


# ── 4. 时间间隔检查 ─────────────────────────────────────────────────────────

def check_delta(subjects):
    print("\n=== 4. 时间间隔检查 ===")
    months_seen = set()
    bad_months  = 0

    for s in subjects.values():
        for v in s["visits"]:
            months_seen.add(v["month"])
            if v["month"] not in Config.VISIT_MONTHS.values():
                bad_months += 1

    check(bad_months == 0,
          f"所有访视月份合法 {sorted(months_seen)}",
          f"{bad_months} 个访视月份不在预设列表中")
    warn_if(months_seen != set(Config.VISIT_MONTHS.values()),
            f"只出现了部分时间点 {sorted(months_seen)}，预期 {sorted(Config.VISIT_MONTHS.values())}")


# ── 5. 图像文件检查 ─────────────────────────────────────────────────────────

def check_images(subjects, n_samples=5, load_content=False):
    print("\n=== 5. 图像文件检查 ===")
    mri_total = mri_found = 0
    pet_total = pet_found = 0
    sample_paths = {"mri": [], "pet": []}

    for s in subjects.values():
        for v in s["visits"]:
            if v["mri_path"] is not None:
                mri_total += 1
                if os.path.exists(v["mri_path"]):
                    mri_found += 1
                    if len(sample_paths["mri"]) < n_samples:
                        sample_paths["mri"].append(v["mri_path"])
            if v["pet_path"] is not None:
                pet_total += 1
                if os.path.exists(v["pet_path"]):
                    pet_found += 1
                    if len(sample_paths["pet"]) < n_samples:
                        sample_paths["pet"].append(v["pet_path"])

    all_visits = sum(len(s["visits"]) for s in subjects.values())
    print(f"  {INFO} MRI: {mri_found}/{mri_total} 文件存在 "
          f"(覆盖 {100*mri_total/all_visits:.1f}% 访视)")
    print(f"  {INFO} PET: {pet_found}/{pet_total} 文件存在 "
          f"(覆盖 {100*pet_total/all_visits:.1f}% 访视)")

    if mri_total > 0:
        check(mri_found == mri_total,
              "所有 mri_path 文件都存在",
              f"{mri_total - mri_found} 个 MRI 路径不存在（会被当作 None 跳过）")
    if pet_total > 0:
        check(pet_found == pet_total,
              "所有 pet_path 文件都存在",
              f"{pet_total - pet_found} 个 PET 路径不存在（会被当作 None 跳过）")

    warn_if(mri_total == 0, "没有任何 MRI 路径，模型将退化为纯表格模型")
    warn_if(pet_total == 0, "没有任何 PET 路径")

    if not load_content:
        print(f"  {INFO} 跳过图像内容检查（加 --load_images 开启）")
        return

    for modality, paths in sample_paths.items():
        print(f"\n  -- {modality.upper()} 图像抽样检查 ({len(paths)} 张) --")
        for p in paths:
            img = np.load(p).astype(np.float32)
            if img.ndim == 3:
                img = img[np.newaxis]
            shape_ok = img.shape == (1,) + Config.TARGET_SHAPE
            vmin, vmax = img.min(), img.max()
            has_nan = np.isnan(img).any()
            has_inf = np.isinf(img).any()
            fname   = os.path.basename(p)
            tag = PASS if (shape_ok and not has_nan and not has_inf) else FAIL
            print(f"    {tag} {fname:40s} shape={img.shape} "
                  f"min={vmin:.3f} max={vmax:.3f} "
                  f"nan={has_nan} inf={has_inf}")
            warn_if(vmax > 100, f"{fname}: 最大值={vmax:.1f}，图像可能未归一化到 [0,1]")
            warn_if(vmin < 0,   f"{fname}: 最小值={vmin:.3f}，图像含负值，请检查归一化")


# ── 6. Dataset 输出检查 ─────────────────────────────────────────────────────

def check_dataset_output(subjects, packed):
    print("\n=== 6. Dataset / collate_fn 输出检查 ===")
    try:
        from dataset import ADNIDataset, collate_fn
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"  {FAIL} 无法导入 dataset.py: {e}")
        return

    ptids = sorted(subjects.keys())[:20]   # 只取前 20 个，快速验证
    ds = ADNIDataset(
        ptids, subjects,
        packed["bio_scaler"], packed["age_scaler"], packed["edu_scaler"],
        load_images=False
    )
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, shuffle=False)
    batch  = next(iter(loader))

    # shape 检查
    B = batch["lengths"].shape[0]
    T = batch["non_img_seq"].shape[1]
    print(f"  {INFO} batch size={B}, T_max={T}")

    expected = {
        "non_img_seq"  : (B, T, Config.NON_IMG_DIM),
        "bio_mask_seq" : (B, T, Config.BIOMARKER_DIM),
        "mod_avail"    : (B, T, 2),
        "delta_seq"    : (B, T, 1),
        "dx_seq"       : (B, T),
    }
    for key, exp_shape in expected.items():
        actual = tuple(batch[key].shape)
        check(actual == exp_shape,
              f"{key}: shape={actual}",
              f"{key}: shape={actual}，预期 {exp_shape}")

    # 数值检查
    ni = batch["non_img_seq"].numpy()
    check(np.isfinite(ni).all(),
          "non_img_seq 无 NaN/Inf",
          f"non_img_seq 含 {np.isnan(ni).sum()} NaN, {np.isinf(ni).sum()} Inf")

    # bio 标准化后应大致在 [-5, 5]
    bio_part = ni[:, :, :Config.BIOMARKER_DIM]
    bio_mask = batch["bio_mask_seq"].numpy()
    obs_vals = bio_part[bio_mask > 0]
    if len(obs_vals) > 0:
        warn_if(np.abs(obs_vals).max() > 10,
                f"z-score 后生物标志物绝对值最大={np.abs(obs_vals).max():.2f}，可能标准化有问题")
        print(f"  {INFO} 生物标志物(z-score后): mean={obs_vals.mean():.3f} "
              f"std={obs_vals.std():.3f} range=[{obs_vals.min():.2f}, {obs_vals.max():.2f}]")

    # mod_avail 应为 0/1
    ma = batch["mod_avail"].numpy()
    check(np.all((ma == 0) | (ma == 1)),
          "mod_avail 只含 0/1",
          f"mod_avail 含非 0/1 值: unique={np.unique(ma)}")

    # delta_seq 第 0 时间步应全为 0
    delta = batch["delta_seq"].numpy()
    check((delta[:, 0, 0] == 0).all(),
          "delta_seq 第 0 步全为 0",
          f"delta_seq 第 0 步不全为 0: {delta[:, 0, 0]}")

    # dx_seq 值域
    dx = batch["dx_seq"].numpy()
    unique_dx = np.unique(dx)
    check(set(unique_dx).issubset({-1, 0, 1, 2}),
          f"dx_seq 值域合法 {unique_dx}",
          f"dx_seq 含非法值 {unique_dx}")


# ── 7. 影像模态可用率汇总 ────────────────────────────────────────────────────

def check_modality_rate(subjects):
    print("\n=== 7. 影像模态可用率（按访视时间点）===")
    from collections import defaultdict
    mri_by_month = defaultdict(lambda: [0, 0])  # [有, 总]
    pet_by_month = defaultdict(lambda: [0, 0])

    for s in subjects.values():
        for v in s["visits"]:
            m = v["month"]
            mri_by_month[m][1] += 1
            pet_by_month[m][1] += 1
            if v["mri_path"] is not None and os.path.exists(v["mri_path"]):
                mri_by_month[m][0] += 1
            if v["pet_path"] is not None and os.path.exists(v["pet_path"]):
                pet_by_month[m][0] += 1

    print(f"  {'月份':>6} {'MRI有/总':>12} {'MRI%':>7} {'PET有/总':>12} {'PET%':>7}")
    print("  " + "-"*50)
    for m in sorted(mri_by_month.keys()):
        mr, mt = mri_by_month[m]
        pr, pt = pet_by_month[m]
        print(f"  {'M'+str(m):>6} {mr:>5}/{mt:<5}    {100*mr/max(mt,1):>5.1f}%  "
              f"{pr:>5}/{pt:<5}    {100*pr/max(pt,1):>5.1f}%")

    total_visits = sum(v[1] for v in mri_by_month.values())
    total_mri    = sum(v[0] for v in mri_by_month.values())
    total_pet    = sum(v[0] for v in pet_by_month.values())
    print(f"\n  {INFO} 总体: MRI={total_mri}/{total_visits}({100*total_mri/max(total_visits,1):.1f}%) "
          f"PET={total_pet}/{total_visits}({100*total_pet/max(total_visits,1):.1f}%)")
    warn_if(total_mri / max(total_visits, 1) < 0.1,
            "MRI 总体覆盖率 < 10%，影像分支几乎不会被激活，请检查预处理路径")
    warn_if(total_pet / max(total_visits, 1) < 0.05,
            "PET 总体覆盖率 < 5%，PET 分支几乎不会被激活")


# ── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",          default=Config.TAB_PROCESSED_PATH)
    parser.add_argument("--load_images",  action="store_true",
                        help="抽样加载图像文件做数值检查（慢）")
    parser.add_argument("--n_img_samples", type=int, default=5,
                        help="每个模态抽查图像数量（默认 5）")
    args = parser.parse_args()

    if not os.path.exists(args.pkl):
        print(f"{FAIL} 找不到 {args.pkl}，请先运行 python data/preprocess_tabular.py")
        sys.exit(1)

    print(f"加载 {args.pkl} ...")
    with open(args.pkl, "rb") as f:
        packed = pickle.load(f)

    subjects = check_pkl(packed)
    check_biomarkers(subjects)
    check_dx(subjects)
    check_delta(subjects)
    check_images(subjects, n_samples=args.n_img_samples, load_content=args.load_images)
    check_dataset_output(subjects, packed)
    check_modality_rate(subjects)

    print("\n" + "="*60)
    print("检查完成。WARN 项不会导致崩溃但可能影响训练质量，FAIL 项需修复。")


if __name__ == "__main__":
    main()
