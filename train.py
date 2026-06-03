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
from typing import Dict, List, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')        # 非交互后端，适合无显示器服务器
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models.lmdp_net import LMDPNet
from dataset import build_dataloaders
from evaluate import evaluate_fold, evaluate_ablation, print_metrics

ABLATION_INTERVAL = 10   # 每隔多少 epoch 做一次消融验证


# ─── 单 Epoch 训练 ────────────────────────────────────────────────────────────

def train_epoch(model: LMDPNet,
                loader,
                optimizer: torch.optim.Optimizer,
                device: str,
                load_images: bool = True,
                scaler=None) -> Dict:
    model.train()
    total_loss = lp_sum = li_sum = lf_sum = 0.0
    recon_sum = kl_sum = 0.0
    c_mri_sum = c_pet_sum = c_prior_sum = 0.0
    c_mri_cond_sum = c_pet_cond_sum = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        try:
            batch = _to_device(batch, device, load_images)
            batch = _mask_current_dx(batch, Config.DX_MASK_PROB)

            # 检查输入数据是否含 NaN/Inf（定位损坏样本）
            for key in ("mri", "pet", "x"):
                val = batch.get(key)
                if val is not None and isinstance(val, torch.Tensor):
                    if not torch.isfinite(val).all():
                        print(f"\n  [数据异常] batch['{key}'] 含 NaN/Inf，"
                              f"batch_idx={n_batches}  "
                              f"nan={torch.isnan(val).sum().item()}  "
                              f"inf={torch.isinf(val).sum().item()}")

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    out = model(batch, is_training=True)
                    loss = out["total_loss"]
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n  [警告] loss={loss.item():.4f}，跳过该 batch")
                    scaler.update()
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(batch, is_training=True)
                loss = out["total_loss"]
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n  [警告] loss={loss.item():.4f}，跳过该 batch")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        except RuntimeError as e:
            err_str = str(e)
            if "out of memory" in err_str:
                print(f"\n  [OOM] {e}\n  释放显存后继续...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            # CUDA context errors are unrecoverable — re-raise immediately
            raise

        total_loss     += loss.item()
        lp_sum         += out["lp"]
        li_sum         += out["li"]
        lf_sum         += out["lf"]
        recon_sum      += out.get("lf_recon",         0.0)
        kl_sum         += out.get("lf_kl",            0.0)
        c_mri_sum      += out.get("contrib_mri",      0.0)
        c_pet_sum      += out.get("contrib_pet",      0.0)
        c_prior_sum    += out.get("contrib_prior",    1.0)
        c_mri_cond_sum += out.get("contrib_mri_cond", 0.0)
        c_pet_cond_sum += out.get("contrib_pet_cond", 0.0)
        n_batches      += 1

    nb = max(n_batches, 1)
    return {
        "loss"             : total_loss / nb,
        "lp"               : lp_sum / nb,
        "li"               : li_sum / nb,
        "lf"               : lf_sum / nb,
        "lf_recon"         : recon_sum / nb,
        "lf_kl"            : kl_sum / nb,
        "contrib_mri"      : c_mri_sum / nb,
        "contrib_pet"      : c_pet_sum / nb,
        "contrib_prior"    : c_prior_sum / nb,
        "contrib_mri_cond" : c_mri_cond_sum / nb,   # 有 MRI 时的编码器真实贡献
        "contrib_pet_cond" : c_pet_cond_sum / nb,   # 有 PET 时的编码器真实贡献
    }


# ─── 贡献率可视化 ─────────────────────────────────────────────────────────────

def _plot_contrib_fold(history: List[Tuple], save_dir: str, fold_idx: int) -> None:
    """
    单折堆叠面积图：x=epoch，y=贡献率，三层分别为 MRI / PET / Prior。
    保存到 <save_dir>/contrib_rates.png
    """
    if not _HAS_MPL or not history:
        return

    epochs  = [h[0] for h in history]
    c_mri   = np.array([h[1] for h in history])
    c_pet   = np.array([h[2] for h in history])
    c_prior = np.array([h[3] for h in history])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(epochs, c_mri, c_pet, c_prior,
                 labels=["MRI", "PET", "Prior  N(0,I)"],
                 colors=["#4C72B0", "#DD8452", "#AAAAAA"],
                 alpha=0.85)
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Contribution Rate", fontsize=12)
    ax.set_title(
        f"Fold {fold_idx + 1} — Modality Contribution Rates  (PoE precision weighting)",
        fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()

    path = os.path.join(save_dir, "contrib_rates.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 模态贡献率曲线 → {path}")


def _plot_contrib_all_folds(all_histories: List[List[Tuple]],
                             save_dir: str) -> None:
    """
    多折汇总图：三个子图分别展示 MRI / PET / Prior 的均值 ± 标准差。
    保存到 <save_dir>/contrib_rates_all_folds.png
    """
    if not _HAS_MPL or not all_histories:
        return

    labels = ["MRI", "PET", "Prior  N(0,I)"]
    colors = ["#4C72B0", "#DD8452", "#888888"]

    # 按最短折对齐
    min_len = min(len(h) for h in all_histories if h)
    if min_len == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for col, (label, color) in enumerate(zip(labels, colors)):
        ax = axes[col]
        mat = np.array([[h[col + 1] for h in hist[:min_len]]
                        for hist in all_histories if hist])  # (n_folds, min_len)
        epochs = [all_histories[0][i][0] for i in range(min_len)]

        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)

        ax.fill_between(epochs, mean - std, mean + std,
                        alpha=0.25, color=color)
        ax.plot(epochs, mean, color=color, linewidth=2.2, label="Mean ± Std")

        # 各折细线
        fold_clrs = plt.cm.tab10.colors
        for fi, row in enumerate(mat):
            ax.plot(epochs, row, color=fold_clrs[fi % 10],
                    linewidth=0.8, alpha=0.55, label=f"Fold {fi+1}")

        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(linestyle='--', alpha=0.35)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Modality Contribution Rates — All Folds  (mean ± std)",
        fontsize=13, y=1.01)
    fig.tight_layout()

    path = os.path.join(save_dir, "contrib_rates_all_folds.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[图] 全折贡献率对比图 → {path}")


def _plot_ablation_fold(history: List[Tuple], save_dir: str, fold_idx: int) -> None:
    """
    消融研究折线图：对比四种模态配置的验证准确率随 epoch 变化。
    折线间距 = 该模态对预测的真实贡献量。
    保存到 <save_dir>/ablation_study.png
    """
    if not _HAS_MPL or not history:
        return

    epochs     = [h[0] for h in history]
    acc_full   = [h[1] for h in history]
    acc_no_mri = [h[2] for h in history]
    acc_no_pet = [h[3] for h in history]
    acc_no_img = [h[4] for h in history]

    fig, (ax, ax_delta) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # ── 上图：绝对准确率 ────────────────────────────────────────────────
    ax.plot(epochs, acc_full,   color="black",    linewidth=2.2, label="Full (MRI+PET+Tab)")
    ax.plot(epochs, acc_no_mri, color="#4C72B0",  linewidth=1.8, linestyle="--", label="No MRI")
    ax.plot(epochs, acc_no_pet, color="#DD8452",  linewidth=1.8, linestyle="--", label="No PET")
    ax.plot(epochs, acc_no_img, color="#888888",  linewidth=1.5, linestyle=":",  label="Tab only")
    ax.fill_between(epochs, acc_no_mri, acc_full, alpha=0.12, color="#4C72B0")
    ax.fill_between(epochs, acc_no_pet, acc_full, alpha=0.12, color="#DD8452")
    ax.set_ylabel("Val Accuracy", fontsize=11)
    ax.set_title(f"Fold {fold_idx+1} — Modality Ablation Study", fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(linestyle="--", alpha=0.4)

    # ── 下图：模态贡献增益 Δacc ─────────────────────────────────────────
    delta_mri = [f - n for f, n in zip(acc_full, acc_no_mri)]
    delta_pet = [f - n for f, n in zip(acc_full, acc_no_pet)]
    delta_img = [f - n for f, n in zip(acc_full, acc_no_img)]
    ax_delta.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_delta.plot(epochs, delta_mri, color="#4C72B0", linewidth=1.8, label="ΔMRI contrib")
    ax_delta.plot(epochs, delta_pet, color="#DD8452", linewidth=1.8, label="ΔPET contrib")
    ax_delta.plot(epochs, delta_img, color="#888888", linewidth=1.5, linestyle=":", label="ΔImage contrib")
    ax_delta.fill_between(epochs, 0, delta_mri, alpha=0.15, color="#4C72B0")
    ax_delta.fill_between(epochs, 0, delta_pet, alpha=0.15, color="#DD8452")
    ax_delta.set_xlabel("Epoch", fontsize=11)
    ax_delta.set_ylabel("Δ Accuracy (gain over ablated)", fontsize=11)
    ax_delta.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_delta.legend(fontsize=10)
    ax_delta.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(save_dir, "ablation_study.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 消融研究图 → {path}")


# ─── 单 Fold 训练 ─────────────────────────────────────────────────────────────

def train_fold(fold_idx:    int,
               args,
               device:      str) -> Tuple[Dict, List[Tuple]]:
    """
    训练第 fold_idx 折，返回 (最佳验证指标, 贡献率历史)。
    贡献率历史格式：[(epoch, c_mri, c_pet, c_prior), ...]
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
        filter_no_image = args.filter_no_image,
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

    optimizer = optim.Adam(model.parameters(),
                           lr=Config.LEARNING_RATE,
                           weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    # 混合精度：float16 激活值，显存减半（仅 CUDA 启用）
    scaler = None if (device != "cuda" or args.no_amp) else torch.amp.GradScaler('cuda')

    best_val_loss     = float("inf")
    best_metrics      = {}
    patience          = 20
    no_improve        = 0
    contrib_history:  List[Tuple] = []
    ablation_history: List[Tuple] = []   # (epoch, acc_full, acc_no_mri, acc_no_pet, acc_no_img)

    ckpt_dir = os.path.join("checkpoints", f"fold{fold_idx}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 保存初始值，fold 结束后恢复——避免多 fold 训练时 warmup 状态污染
    _original_kl_weight = Config.KL_WEIGHT

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        t0 = time.time()

        # KL warmup：线性从 0 增长到目标 kl_weight，仅修改本 fold 的临时值
        if args.kl_warmup_epochs > 0:
            Config.KL_WEIGHT = args.kl_weight * min(1.0, epoch / args.kl_warmup_epochs)
        else:
            Config.KL_WEIGHT = args.kl_weight

        train_log = train_epoch(model, train_loader, optimizer, device, args.load_images, scaler)
        if device == "cuda":
            torch.cuda.empty_cache()
        val_metrics = evaluate_fold(model, val_loader, device, args.load_images)

        scheduler.step()
        elapsed = time.time() - t0
        if device == "cuda":
            mem_used = torch.cuda.memory_reserved() / 1024**3
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        contrib_history.append((
            epoch,
            train_log["contrib_mri"],
            train_log["contrib_pet"],
            train_log["contrib_prior"],
        ))

        # ── 消融验证（每 ABLATION_INTERVAL 轮一次）──────────────────────
        if args.load_images and epoch % ABLATION_INTERVAL == 0:
            acc_full   = val_metrics.get("acc", 0.0)
            acc_no_mri = evaluate_ablation(model, val_loader, device,
                                           load_images=True, no_mri=True)
            acc_no_pet = evaluate_ablation(model, val_loader, device,
                                           load_images=True, no_pet=True)
            acc_no_img = evaluate_ablation(model, val_loader, device,
                                           load_images=True, no_mri=True, no_pet=True)
            ablation_history.append((epoch, acc_full, acc_no_mri, acc_no_pet, acc_no_img))
            print(f"  [消融] "
                  f"full={acc_full:.3f}  "
                  f"no_MRI={acc_no_mri:.3f}(Δ{acc_full-acc_no_mri:+.3f})  "
                  f"no_PET={acc_no_pet:.3f}(Δ{acc_full-acc_no_pet:+.3f})  "
                  f"tab_only={acc_no_img:.3f}(Δ{acc_full-acc_no_img:+.3f})")

        # 格式：MRI=17.8%(↑58%有图时) 表示整体均摊贡献 vs 有 MRI 时的真实编码器信息量
        c_mri_cond = train_log.get("contrib_mri_cond", 0.0)
        c_pet_cond = train_log.get("contrib_pet_cond", 0.0)
        mem_str = f" GPU={mem_used:.1f}/{mem_total:.0f}GB" if device == "cuda" else ""
        step_str = ""
        for t, sm in sorted(val_metrics.get("step_metrics", {}).items()):
            step_str += f" t{t}:{sm['acc']:.3f}(n={sm['n']})"

        print(f"  Epoch {epoch:3d}/{Config.NUM_EPOCHS} "
              f"| loss={train_log['loss']:.4f} "
              f"lp={train_log['lp']:.4f} "
              f"li={train_log['li']:.4f} "
              f"lf={train_log['lf']:.4f}"
              f"(rec={train_log['lf_recon']:.4f} kl={train_log['lf_kl']:.2f}) "
              f"| MRI={train_log['contrib_mri']:.1%}(↑{c_mri_cond:.0%}) "
              f"PET={train_log['contrib_pet']:.1%}(↑{c_pet_cond:.0%}) "
              f"prior={train_log['contrib_prior']:.1%} "
              f"| val_acc={val_metrics.get('acc', 0):.4f} "
              f"mAUC={val_metrics.get('mauc', 0):.4f} "
              f"| steps:{step_str}"
              f"| {elapsed:.1f}s{mem_str}")

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

    # 恢复 Config.KL_WEIGHT，避免下一个 fold 的 warmup 起点被污染
    Config.KL_WEIGHT = _original_kl_weight

    print(f"\n  Fold {fold_idx + 1} 最佳结果:")
    print_metrics(best_metrics)
    _plot_contrib_fold(contrib_history, ckpt_dir, fold_idx)
    _plot_ablation_fold(ablation_history, ckpt_dir, fold_idx)
    return best_metrics, contrib_history


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
    parser.add_argument("--kl_weight",   type=float, default=Config.KL_WEIGHT,
                        help="β-VAE KL 权重（论文推荐范围 0.001–0.1）")
    parser.add_argument("--filter_no_image", action="store_true",
                        help="过滤掉所有访次均无图像的受试者")
    parser.add_argument("--kl_warmup_epochs", type=int, default=20,
                        help="KL 权重从 0 线性增长到 --kl_weight 所需的 epoch 数（0=不做 warmup）")
    parser.add_argument("--no_amp", action="store_true",
                        help="禁用混合精度（AMP），用 FP32 训练，排查 CUDA 数值问题")
    args = parser.parse_args()
    args.load_images = not args.no_images
    Config.KL_WEIGHT = args.kl_weight

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

    all_fold_metrics  = []
    all_fold_histories = []
    for fold_idx in fold_range:
        metrics, history = train_fold(fold_idx, args, device)
        all_fold_metrics.append(metrics)
        all_fold_histories.append(history)

    if len(all_fold_metrics) > 1:
        print(f"\n{'='*60}")
        print(f"  {Config.K_FOLDS}-Fold 平均结果")
        print(f"{'='*60}")
        keys = ["acc", "pre", "rec", "mauc"]
        for k in keys:
            vals = [m.get(k, 0) for m in all_fold_metrics]
            print(f"  {k.upper():6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        _plot_contrib_all_folds(all_fold_histories,
                                save_dir=os.path.join("checkpoints"))


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
        # Zero out image availability so _encode_modality treats all samples
        # as image-free; prevents phantom contributions from logvar=0 default.
        if "mod_avail" in result:
            result["mod_avail"] = torch.zeros_like(result["mod_avail"])
    return result


if __name__ == "__main__":
    main()
