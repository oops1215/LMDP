"""
LMDP-Net 训练脚本
训练主脚本（5-fold、早停、KL warmup、消融评估）
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

  # 从最近一次 checkpoint 续训
  python train.py --fold 0 --resume checkpoints/fold0/last_model.pth
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
from evaluate import (evaluate_fold, evaluate_ablation, evaluate_image_gain,
                      print_metrics)


# ─── 单 Epoch 训练 ────────────────────────────────────────────────────────────

def train_epoch(model: LMDPNet,
                loader,
                optimizer: torch.optim.Optimizer,
                device: str,
                load_images: bool = True,
                scaler=None,
                grad_accum_steps: int = 1) -> Dict:
    model.train()
    total_loss = lp_sum = li_sum = lf_sum = laux_sum = 0.0
    recon_sum = kl_sum = gate_sum = 0.0
    c_mri_sum = c_pet_sum = c_prior_sum = 0.0
    c_mri_cond_sum = c_pet_cond_sum = 0.0
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        batch = _to_device(batch, device, load_images)
        batch = _mask_current_dx(batch, Config.DX_MASK_PROB)

        is_accum_step = ((batch_idx + 1) % grad_accum_steps == 0)
        is_last_batch = (batch_idx == len(loader) - 1)

        try:
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    out = model(batch, is_training=True)
                    loss = out["total_loss"] / grad_accum_steps
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n  [警告] loss={loss.item():.4f}，跳过该 batch")
                    if is_accum_step or is_last_batch:
                        scaler.update()
                    continue
                scaler.scale(loss).backward()
                if is_accum_step or is_last_batch:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                out = model(batch, is_training=True)
                loss = out["total_loss"] / grad_accum_steps
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n  [警告] loss={loss.item():.4f}，跳过该 batch")
                    continue
                loss.backward()
                if is_accum_step or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA error" in str(e):
                print(f"\n  [OOM/CUDA错误] {e}\n  释放显存后继续...")
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue
            raise

        total_loss     += loss.item() * grad_accum_steps   # 还原为未缩放的 loss 用于日志
        lp_sum         += out["lp"]
        li_sum         += out["li"]
        lf_sum         += out["lf"]
        laux_sum       += out.get("l_aux",            0.0)
        recon_sum      += out.get("lf_recon",         0.0)
        kl_sum         += out.get("lf_kl",            0.0)
        gate_sum       += out.get("gate_mean",        0.0)
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
        "l_aux"            : laux_sum / nb,
        "lf_recon"         : recon_sum / nb,
        "lf_kl"            : kl_sum / nb,
        "gate_mean"        : gate_sum / nb,
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
        filter_mri_subjects = args.filter_mri_subjects,
        filter_matched_visits = args.filter_matched_visits,
        image_cache_size = args.image_cache_size,
    )

    # latent_dim 始终保持 Config.LATENT_DIM；
    # 不使用图像时 M3VAE 的 fused_mu 会自动返回全零，不影响架构
    model = LMDPNet(
        latent_dim     = Config.LATENT_DIM,
        hidden_dim     = Config.HIDDEN_DIM,
        non_img_dim    = Config.NON_IMG_DIM,
        num_classes    = Config.NUM_CLASSES,
        biomarker_dim  = Config.BIOMARKER_DIM,
        use_checkpoint = Config.USE_CHECKPOINT,
    ).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=Config.LEARNING_RATE,
                           weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.LR_FACTOR,
        patience=Config.LR_PATIENCE,
        min_lr=Config.MIN_LR,
    )
    # 混合精度：float16 激活值，显存减半（仅 CUDA 启用）
    scaler = None if (device != "cuda" or args.no_amp) else torch.amp.GradScaler('cuda')

    best_val_loss     = float("inf")
    best_mauc         = -float("inf")
    best_select_score = -float("inf")
    best_metrics      = {}
    patience          = Config.EARLY_STOP_PATIENCE
    no_improve        = 0
    start_epoch       = 1
    contrib_history:  List[Tuple] = []
    ablation_history: List[Tuple] = []   # (epoch, acc_full, acc_no_mri, acc_no_pet, acc_no_img)
    imggain_history:  List[Dict]  = []   # 每次消融的真实图像增益记录（dict）

    ckpt_dir = os.path.join("checkpoints", f"fold{fold_idx}")
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.resume:
        resume_path = args.resume
        if args.fold == -1 and "{fold}" in resume_path:
            resume_path = resume_path.format(fold=fold_idx)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"找不到 resume checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        best_metrics = ckpt.get("best_metrics", ckpt.get("metrics", {}))
        best_val_loss = ckpt.get("best_val_loss", best_metrics.get("val_loss", best_val_loss))
        best_mauc = ckpt.get("best_mauc", best_metrics.get("mauc", best_mauc))
        best_select_score = ckpt.get("best_select_score", best_metrics.get("select_score", best_select_score))
        no_improve = ckpt.get("no_improve", 0)
        contrib_history = ckpt.get("contrib_history", [])
        ablation_history = ckpt.get("ablation_history", [])
        imggain_history = ckpt.get("imggain_history", [])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  [resume] 从 {resume_path} 继续: epoch {start_epoch}/{Config.NUM_EPOCHS}")
        if start_epoch > Config.NUM_EPOCHS:
            print(f"  [resume] checkpoint 已完成到 epoch {start_epoch - 1}，无需继续训练")

    # 保存初始值，fold 结束后恢复——避免多 fold 训练时 warmup 状态污染
    _original_kl_weight = Config.KL_WEIGHT

    for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
        t0 = time.time()

        # KL warmup：线性从 0 增长到目标 kl_weight，仅修改本 fold 的临时值
        if args.kl_warmup_epochs > 0:
            Config.KL_WEIGHT = args.kl_weight * min(1.0, epoch / args.kl_warmup_epochs)
        else:
            Config.KL_WEIGHT = args.kl_weight

        train_log = train_epoch(model, train_loader, optimizer, device,
                                args.load_images, scaler,
                                grad_accum_steps=Config.GRAD_ACCUM_STEPS)
        if device == "cuda":
            torch.cuda.empty_cache()
        val_metrics = evaluate_fold(model, val_loader, device, args.load_images)

        current_mauc = val_metrics.get("mauc", 0.0)
        if not np.isfinite(current_mauc):
            print("  [警告] 验证 mAUC 为 NaN/Inf，scheduler 与选模临时按 0.0 处理")
            current_mauc = 0.0
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(current_mauc)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < old_lr:
            print(f"  [LR] mAUC 停滞，学习率 {old_lr:.2e} → {current_lr:.2e}")
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
        if args.load_images and args.ablation_interval > 0 and epoch % args.ablation_interval == 0:
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

            # ── 真实图像增益（只看源步有图的样本，这是真正的优化目标）──────
            gain = evaluate_image_gain(model, val_loader, device)
            gain["epoch"] = epoch
            imggain_history.append(gain)
            print(f"  [图像增益] 有图样本(n={gain['n_img']}): "
                  f"full={gain['acc_full_img']:.3f} tab={gain['acc_tab_img']:.3f} "
                  f"ΔACC={gain['delta_img']:+.3f} "
                  f"mAUC={gain['mauc_full_img']:.3f}/{gain['mauc_tab_img']:.3f} "
                  f"ΔmAUC={gain['delta_img_mauc']:+.3f}  "
                  f"| 无图样本(n={gain['n_noimg']}): full={gain['acc_full_noimg']:.3f}")

        # 格式：MRI=17.8%(↑58%有图时) 表示整体均摊贡献 vs 有 MRI 时的真实编码器信息量
        c_mri_cond = train_log.get("contrib_mri_cond", 0.0)
        c_pet_cond = train_log.get("contrib_pet_cond", 0.0)
        mem_str = f" GPU={mem_used:.1f}/{mem_total:.0f}GB" if device == "cuda" else ""
        from config import Config as _C
        step_acc = val_metrics.get("step_acc", {})
        step_n   = val_metrics.get("step_n",   {})
        steps_str = " ".join(
            f"t{t}:{step_acc[t]:.3f}(n={step_n[t]})"
            for t in sorted(step_acc.keys())
        )
        print(f"  Epoch {epoch:3d}/{Config.NUM_EPOCHS} "
              f"| loss={train_log['loss']:.4f} "
              f"lp={train_log['lp']:.4f} "
              f"li={train_log['li']:.4f} "
              f"lf={train_log['lf']:.4f}"
              f"(wrec={train_log['lf_recon']*_C.RECON_WEIGHT:.4f} kl={train_log['lf_kl']:.2f}) "
              f"laux={train_log.get('l_aux', 0):.4f} "
              f"gate={train_log.get('gate_mean', 0):.3f} "
              f"| MRI={train_log['contrib_mri']:.1%}(↑{c_mri_cond:.0%}) "
              f"PET={train_log['contrib_pet']:.1%}(↑{c_pet_cond:.0%}) "
              f"prior={train_log['contrib_prior']:.1%} "
              f"| val_acc={val_metrics.get('acc', 0):.4f} "
              f"mAUC={val_metrics.get('mauc', 0):.4f} "
              f"lr={current_lr:.2e} "
              f"| steps: {steps_str}"
              f"| {elapsed:.1f}s{mem_str}")

        # 早停与模型保存：以 mAUC + 有图样本 ΔmAUC 为选择指标，
        # 防止全体验证 mAUC 提升但模型实际继续忽略图像。
        val_loss = val_metrics.get("val_loss", float("inf"))
        # 选模默认以验证 mAUC 为主。图像增益只在刚完成消融评估的 epoch 加入，
        # 避免后续 epoch 反复使用过期的 delta_img_mauc 误导早停/保存。
        latest_gain = imggain_history[-1] if imggain_history and imggain_history[-1].get("epoch") == epoch else {}
        delta_img_mauc = latest_gain.get("delta_img_mauc", 0.0)
        if not np.isfinite(delta_img_mauc):
            delta_img_mauc = 0.0
        select_score = current_mauc + Config.IMG_GAIN_WEIGHT * delta_img_mauc
        val_metrics["select_score"] = select_score
        val_metrics["delta_img_mauc"] = delta_img_mauc
        if select_score > best_select_score + Config.EARLY_STOP_MIN_DELTA:
            best_select_score = select_score
            best_mauc = current_mauc
            best_val_loss = val_loss
            best_metrics  = val_metrics.copy()
            no_improve    = 0
            torch.save({
                "epoch"     : epoch,
                "model_state": model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "scheduler" : scheduler.state_dict(),
                "scaler"    : scaler.state_dict() if scaler is not None else None,
                "metrics"   : best_metrics,
                "best_metrics": best_metrics,
                "best_val_loss": best_val_loss,
                "best_mauc": best_mauc,
                "best_select_score": best_select_score,
                "no_improve": no_improve,
                "contrib_history": contrib_history,
                "ablation_history": ablation_history,
                "imggain_history": imggain_history,
            }, os.path.join(ckpt_dir, "best_model.pth"))
        else:
            no_improve += 1

        torch.save({
            "epoch"     : epoch,
            "model_state": model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "scheduler" : scheduler.state_dict(),
            "scaler"    : scaler.state_dict() if scaler is not None else None,
            "metrics"   : val_metrics,
            "best_metrics": best_metrics,
            "best_val_loss": best_val_loss,
            "best_mauc": best_mauc,
            "best_select_score": best_select_score,
            "no_improve": no_improve,
            "contrib_history": contrib_history,
            "ablation_history": ablation_history,
            "imggain_history": imggain_history,
        }, os.path.join(ckpt_dir, "last_model.pth"))

        if no_improve >= patience:
            print(f"  [早停] 选择分数连续 {patience} 轮无改善，"
                  f"最佳 mAUC={best_mauc:.4f} select={best_select_score:.4f}")
            break

    # 恢复 Config.KL_WEIGHT，避免下一个 fold 的 warmup 起点被污染
    Config.KL_WEIGHT = _original_kl_weight

    print(f"\n  Fold {fold_idx + 1} 最佳结果:")
    print_metrics(best_metrics)
    _plot_contrib_fold(contrib_history, ckpt_dir, fold_idx)
    _plot_ablation_fold(ablation_history, ckpt_dir, fold_idx)
    _dump_fold_json(ckpt_dir, fold_idx, best_metrics,
                    contrib_history, ablation_history, imggain_history)
    return best_metrics, contrib_history


# ─── 指标落盘（json）─────────────────────────────────────────────────────────

def _dump_fold_json(save_dir: str,
                    fold_idx: int,
                    best_metrics: Dict,
                    contrib_history: List[Tuple],
                    ablation_history: List[Tuple],
                    imggain_history: List[Dict]) -> None:
    """
    将本折的最佳指标 + 贡献率/消融/图像增益历史写入 <save_dir>/metrics.json。
    取代"只存 png 需反解 checkpoint 才能看数字"的旧流程，
    其中 imggain_history 的 delta_img 是替代贡献率 proxy 的真实优化目标。
    """
    import json

    def _clean(d):
        # 过滤掉非 json 可序列化的值（如 numpy 类型、嵌套 dict 中的 int key）
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[str(k)] = {str(kk): (float(vv) if isinstance(vv, (int, float)) else vv)
                               for kk, vv in v.items()}
            elif isinstance(v, (int, float, str)) or v is None:
                out[str(k)] = v
            else:
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    out[str(k)] = str(v)
        return out

    payload = {
        "fold"            : fold_idx,
        "best_metrics"    : _clean(best_metrics),
        "contrib_history" : [
            {"epoch": e, "c_mri": cm, "c_pet": cp, "c_prior": cpr}
            for (e, cm, cp, cpr) in contrib_history
        ],
        "ablation_history": [
            {"epoch": e, "acc_full": af, "acc_no_mri": anm,
             "acc_no_pet": anp, "acc_no_img": ani}
            for (e, af, anm, anp, ani) in ablation_history
        ],
        "imggain_history" : [_clean(g) for g in imggain_history],
    }

    path = os.path.join(save_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  [json] 指标历史 → {path}")


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LMDP-Net Training")
    parser.add_argument("--fold",        type=int, default=-1,
                        help="指定运行的 fold（-1 = 全部 K 个）")
    parser.add_argument("--no_images",   action="store_true",
                        help="不使用图像特征（只用表格数据，快速调试）")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device",      type=str, default=Config.DEVICE)
    parser.add_argument("--kl_weight",   type=float, default=Config.KL_WEIGHT,
                        help="β-VAE KL 权重（论文推荐范围 0.001–0.1）")
    parser.add_argument("--filter_no_image", action="store_true",
                        help="过滤掉所有访次均无图像的受试者")
    parser.add_argument("--filter_mri_subjects", action="store_true",
                        help="只保留至少有一次 MRI 的受试者，但保留其所有访视（论文式 MRI cohort）")
    parser.add_argument("--filter_matched_visits", action="store_true",
                        help="只保留既有Tabular又有MRI的访视（visit-level 100% MRI，样本更少）")
    parser.add_argument("--image_cache_size", type=int, default=Config.IMAGE_CACHE_SIZE,
                        help="每个 DataLoader worker 缓存的 .npy 图像数量；0=关闭")
    parser.add_argument("--ablation_interval", type=int, default=Config.ABLATION_INTERVAL,
                        help="每隔多少 epoch 做一次消融/图像增益验证；0=关闭")
    parser.add_argument("--kl_warmup_epochs", type=int, default=20,
                        help="KL 权重从 0 线性增长到 --kl_weight 所需的 epoch 数（0=不做 warmup）")
    parser.add_argument("--no_amp",       action="store_true",
                        help="禁用混合精度（AMP），用 FP32 训练，排查 CUDA 数值问题")
    parser.add_argument("--batch_size",   type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--grad_accum",   type=int, default=Config.GRAD_ACCUM_STEPS)
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="禁用梯度检查点（更快但显存占用更大）")
    parser.add_argument("--resume", type=str, default="",
                        help="从 checkpoint 续训；多 fold 可用 checkpoints/fold{fold}/last_model.pth")
    args = parser.parse_args()
    args.load_images = not args.no_images
    Config.KL_WEIGHT        = args.kl_weight
    Config.BATCH_SIZE       = args.batch_size
    Config.GRAD_ACCUM_STEPS = args.grad_accum
    Config.USE_CHECKPOINT   = not args.no_checkpoint

    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
        torch.backends.cudnn.benchmark    = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32   = True

    device = args.device

    # ── 终端日志保存：同时输出到终端和 checkpoints/train_log.txt ──────────
    os.makedirs("checkpoints", exist_ok=True)
    log_path = os.path.join("checkpoints", "train_log.txt")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    log_file.write(f"\n{'='*60}\n"
                   f"训练开始: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"命令: {' '.join(sys.argv)}\n"
                   f"{'='*60}\n")
    sys.stdout = Tee(sys.__stdout__, log_file)

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

    # ── 恢复 stdout，关闭日志文件 ──────────────────────────────────────────
    log_file.close()
    sys.stdout = sys.__stdout__
    print(f"训练日志已保存到: {log_path}")


# ─── 辅助 ────────────────────────────────────────────────────────────────────

class Tee:
    """把 stdout 同时写到终端和日志文件，方便事后检查训练过程。"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()


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
            result[k] = v.to(device, non_blocking=True)
        else:
            result[k] = v
    if not load_images:
        result.pop("mri_seq", None)
        result.pop("pet_seq", None)
    return result


if __name__ == "__main__":
    main()
