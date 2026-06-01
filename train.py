"""
LMDP-Net 训练脚本

论文设置（Section IV.A）：
  - 5-fold 交叉验证
  - Adam 优化器，lr=0.002
  - 初始隐藏状态 h0 = 0
  - 当前访视诊断被随机掩码
  - 等权重损失：L_total = L_p + L_i + L_f
  - 硬件：Intel Xeon Gold 6326 + Nvidia RTX A6000

使用方法：
  # 只使用表格特征（无图像，调试用）
  python train.py --no_images

  # 完整训练（需要预处理后的图像）
  python train.py

  # 指定 fold
  python train.py --fold 0
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models.lmdp_net import LMDPNet
from dataset import build_dataloaders
from evaluate import evaluate_fold, print_metrics


# ─── 单 Epoch 训练 ────────────────────────────────────────────────────────────

def train_epoch(model: LMDPNet,
                loader,
                optimizer: torch.optim.Optimizer,
                device: str,
                load_images: bool = True) -> Dict:
    model.train()
    total_loss = lp_sum = li_sum = lf_sum = 0.0
    c_mri_sum = c_pet_sum = c_prior_sum = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        batch = _to_device(batch, device, load_images)
        batch = _mask_current_dx(batch, Config.DX_MASK_PROB)
        optimizer.zero_grad()

        out = model(batch, is_training=True)
        loss = out["total_loss"]
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        lp_sum     += out["lp"]
        li_sum     += out["li"]
        lf_sum     += out["lf"]
        c_mri_sum  += out.get("contrib_mri",   0.0)
        c_pet_sum  += out.get("contrib_pet",   0.0)
        c_prior_sum += out.get("contrib_prior", 1.0)
        n_batches  += 1

    nb = max(n_batches, 1)
    return {
        "loss"         : total_loss / nb,
        "lp"           : lp_sum / nb,
        "li"           : li_sum / nb,
        "lf"           : lf_sum / nb,
        "contrib_mri"  : c_mri_sum / nb,
        "contrib_pet"  : c_pet_sum / nb,
        "contrib_prior": c_prior_sum / nb,
    }


# ─── 单 Fold 训练 ─────────────────────────────────────────────────────────────

def train_fold(fold_idx:    int,
               args,
               device:      str) -> Dict:
    """
    训练第 fold_idx 折，返回该折的最佳验证指标。
    """
    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx + 1} / {Config.K_FOLDS}")
    print(f"{'='*60}")

    train_loader, val_loader = build_dataloaders(
        processed_data_path = Config.TAB_PROCESSED_PATH,
        k_fold      = Config.K_FOLDS,
        fold_idx    = fold_idx,
        batch_size  = Config.BATCH_SIZE,
        load_images = args.load_images,
        num_workers = args.num_workers,
        seed        = Config.SEED,
    )

    # latent_dim 始终保持 Config.LATENT_DIM；
    # 不使用图像时 M3VAE 的 fused_mu 会自动返回全零，不影响架构
    model = LMDPNet(
        latent_dim    = Config.LATENT_DIM,
        hidden_dim    = Config.HIDDEN_DIM,
        non_img_dim   = Config.NON_IMG_DIM,
        num_classes   = Config.NUM_CLASSES,
        biomarker_dim = Config.BIOMARKER_DIM,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_val_loss = float("inf")
    best_metrics  = {}
    patience      = 20
    no_improve    = 0

    ckpt_dir = os.path.join("checkpoints", f"fold{fold_idx}")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        t0 = time.time()

        train_log = train_epoch(model, train_loader, optimizer, device, args.load_images)
        val_metrics = evaluate_fold(model, val_loader, device, args.load_images)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:3d}/{Config.NUM_EPOCHS} "
              f"| loss={train_log['loss']:.4f} "
              f"(lp={train_log['lp']:.4f} li={train_log['li']:.4f} lf={train_log['lf']:.4f}) "
              f"| contrib MRI={train_log['contrib_mri']:.1%} "
              f"PET={train_log['contrib_pet']:.1%} "
              f"prior={train_log['contrib_prior']:.1%} "
              f"| val_acc={val_metrics.get('acc', 0):.4f} "
              f"mAUC={val_metrics.get('mauc', 0):.4f} "
              f"| {elapsed:.1f}s")

        # 早停与模型保存
        val_loss = val_metrics.get("val_loss", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = val_metrics
            no_improve    = 0
            torch.save({
                "epoch"     : epoch,
                "model_state": model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "metrics"   : best_metrics,
            }, os.path.join(ckpt_dir, "best_model.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [早停] {patience} 轮无改善")
                break

    print(f"\n  Fold {fold_idx + 1} 最佳结果:")
    print_metrics(best_metrics)
    return best_metrics


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LMDP-Net Training")
    parser.add_argument("--fold",        type=int, default=-1,
                        help="指定运行的 fold（-1 = 全部 K 个）")
    parser.add_argument("--no_images",   action="store_true",
                        help="不使用图像特征（只用表格数据，快速调试）")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device",      type=str,
                        default=Config.DEVICE)
    args = parser.parse_args()
    args.load_images = not args.no_images

    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    device = args.device
    print(f"使用设备: {device}")

    if not os.path.exists(Config.TAB_PROCESSED_PATH):
        print(f"[错误] 找不到 {Config.TAB_PROCESSED_PATH}")
        print("请先运行: python data/preprocess_tabular.py")
        sys.exit(1)

    fold_range = range(Config.K_FOLDS) if args.fold == -1 else [args.fold]

    all_fold_metrics = []
    for fold_idx in fold_range:
        metrics = train_fold(fold_idx, args, device)
        all_fold_metrics.append(metrics)

    if len(all_fold_metrics) > 1:
        print(f"\n{'='*60}")
        print(f"  {Config.K_FOLDS}-Fold 平均结果")
        print(f"{'='*60}")
        keys = ["acc", "pre", "rec", "mauc"]
        for k in keys:
            vals = [m.get(k, 0) for m in all_fold_metrics]
            print(f"  {k.upper():6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


# ─── 辅助 ────────────────────────────────────────────────────────────────────

def _mask_current_dx(batch: dict, prob: float) -> dict:
    """
    训练时随机掩码当前访视诊断标签（论文 Section IV.A）。
    每个有效标签以概率 prob 被设为 -1，_compute_lp 会跳过 -1 标签。
    防止模型直接复制当前诊断而不学习纵向变化规律。
    """
    if prob <= 0.0:
        return batch
    dx = batch["dx_seq"].clone()                      # (B, T)
    valid = dx >= 0                                    # 只掩码有效标签
    mask  = torch.rand_like(dx.float()) < prob        # 随机掩码矩阵
    dx[valid & mask] = -1
    return {**batch, "dx_seq": dx}


def _to_device(batch: dict, device: str, load_images: bool) -> dict:
    """将 batch 中的张量移到指定设备。"""
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        else:
            result[k] = v
    if not load_images:
        result.pop("mri_seq", None)
        result.pop("pet_seq", None)
    return result


if __name__ == "__main__":
    main()
