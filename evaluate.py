"""
评估指标与评估函数

论文使用的评估指标（Section IV.A）：
  - 图像重建：MSE, PSNR
  - 生物标志物插补：MAE, MRE
  - 诊断预测：Accuracy, Precision, Recall, mAUC（macro-averaged ROC AUC）

evaluate_fold()：在验证集上完整评估 LMDP-Net。
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              roc_auc_score, confusion_matrix)
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config


# ─── 图像重建指标 ─────────────────────────────────────────────────────────────

def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def compute_psnr(pred: np.ndarray, target: np.ndarray,
                 data_range: float = 1.0) -> float:
    mse = compute_mse(pred, target)
    if mse < 1e-10:
        return float("inf")
    return float(10 * math.log10(data_range ** 2 / mse))


# ─── 生物标志物插补指标 ───────────────────────────────────────────────────────

def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def compute_mre(pred: np.ndarray, target: np.ndarray,
                eps: float = 1e-8) -> float:
    return float(np.mean(np.abs(pred - target) / (np.abs(target) + eps)))


# ─── 诊断预测指标 ─────────────────────────────────────────────────────────────

def compute_classification_metrics(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    y_prob: np.ndarray,
                                    num_classes: int = Config.NUM_CLASSES) -> Dict:
    """
    y_true : (N,) int 真实标签
    y_pred : (N,) int 预测标签
    y_prob : (N, C) float 预测概率

    返回：acc, pre, rec, mauc
    """
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # mAUC：macro-averaged one-vs-rest ROC AUC
    try:
        classes_present = np.unique(y_true)
        if len(classes_present) < 2:
            mauc = float("nan")
        elif len(classes_present) < num_classes:
            # 只对出现的类别计算 AUC
            idx = np.isin(np.arange(num_classes), classes_present)
            mauc = roc_auc_score(
                y_true, y_prob[:, idx],
                multi_class="ovr", average="macro",
                labels=classes_present,
            )
        else:
            mauc = roc_auc_score(y_true, y_prob,
                                  multi_class="ovr", average="macro")
    except Exception:
        mauc = float("nan")
    if not np.isfinite(mauc):
        # 验证 fold 类别缺失或 sklearn AUC 失败时，避免 NaN 继续污染 LR scheduler/选模。
        mauc = 0.0

    return {"acc": acc, "pre": pre, "rec": rec, "mauc": mauc}


# ─── 完整 Fold 评估 ───────────────────────────────────────────────────────────

def evaluate_fold(model,
                  loader,
                  device: str,
                  load_images: bool = True) -> Dict:
    """
    在给定 DataLoader 上评估 LMDP-Net，返回所有指标。
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    all_bio_true, all_bio_pred, all_bio_mask = [], [], []
    total_loss = 0.0
    n_batches  = 0
    # per-step 统计：step_correct[t] / step_total[t]
    step_correct = {}
    step_total   = {}

    with torch.no_grad():
        for batch in loader:
            # 移到设备
            batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}
            if not load_images:
                batch_d.pop("mri_seq", None)
                batch_d.pop("pet_seq", None)

            out = model(batch_d, is_training=True)   # 仍需要 loss
            total_loss += out["total_loss"].item()
            n_batches  += 1

            dx_logits = out["dx_preds"]   # (B, T, C)
            bio_preds = out["bio_preds"]  # (B, T, 6)
            dx_seq    = batch_d["dx_seq"]        # (B, T)
            bio_obs   = batch_d["non_img_seq"][:, :, :Config.BIOMARKER_DIM]
            bio_mask  = batch_d["bio_mask_seq"]  # (B, T, 6)
            lengths   = batch_d["lengths"]        # (B,)

            B, T, C = dx_logits.shape

            for b in range(B):
                L = int(lengths[b].item())
                # 用 t 步预测 t+1 步
                for t in range(min(L - 1, T - 1)):
                    label = int(dx_seq[b, t + 1].item())
                    if label < 0:
                        continue
                    prob = F.softmax(dx_logits[b, t], dim=0).cpu().numpy()
                    pred = int(prob.argmax())
                    all_true.append(label)
                    all_pred.append(pred)
                    all_prob.append(prob)
                    # per-step 统计（t 表示预测源时间步）
                    step_correct[t] = step_correct.get(t, 0) + int(pred == label)
                    step_total[t]   = step_total.get(t, 0) + 1

                # 生物标志物插补评估（仅对实测值）
                for t in range(min(L, T)):
                    m = bio_mask[b, t].cpu().numpy()  # (6,)
                    obs_idx = m > 0
                    if obs_idx.any():
                        bp = bio_preds[b, t].cpu().numpy()[obs_idx]
                        bt = bio_obs[b, t].cpu().numpy()[obs_idx]
                        all_bio_pred.append(bp)
                        all_bio_true.append(bt)

    metrics = {"val_loss": total_loss / max(n_batches, 1)}

    # 诊断预测指标
    if all_true:
        y_true = np.array(all_true)
        y_pred = np.array(all_pred)
        y_prob = np.array(all_prob)
        cls_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        metrics.update(cls_metrics)

    # per-step 准确率
    metrics["step_acc"] = {t: step_correct[t] / step_total[t]
                           for t in sorted(step_total.keys())}
    metrics["step_n"]   = {t: step_total[t] for t in sorted(step_total.keys())}

    # 生物标志物插补指标
    if all_bio_true:
        bp = np.concatenate(all_bio_pred)
        bt = np.concatenate(all_bio_true)
        metrics["mae"] = compute_mae(bp, bt)
        metrics["mre"] = compute_mre(bp, bt)

    return metrics


# ─── 消融评估（快速，仅返回准确率）──────────────────────────────────────────

def evaluate_ablation(model,
                      loader,
                      device: str,
                      load_images: bool = True,
                      no_mri: bool = False,
                      no_pet: bool = False) -> float:
    """
    强制关闭指定模态后评估验证集准确率。

    no_mri=True : 将 mod_avail[:,:,0] 置 0，MRI 编码器输出被忽略
    no_pet=True : 将 mod_avail[:,:,1] 置 0，PET 编码器输出被忽略
    两者同时为 True : 仅用表格特征（相当于无图像基线）

    返回 accuracy (float 0-1)。
    """
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in loader:
            batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}

            # 移除图像张量（节省显存；avail 置 0 已足够让编码器跳过）
            if not load_images or no_mri:
                batch_d.pop("mri_seq", None)
            if not load_images or no_pet:
                batch_d.pop("pet_seq", None)

            # 强制关闭对应模态可用性
            if (no_mri or no_pet) and "mod_avail" in batch_d:
                ma = batch_d["mod_avail"].clone()
                if no_mri:
                    ma[:, :, 0] = 0
                if no_pet:
                    ma[:, :, 1] = 0
                batch_d["mod_avail"] = ma

            out = model(batch_d, is_training=False)
            dx_logits = out["dx_preds"]   # (B, T, C)
            dx_seq    = batch_d["dx_seq"]
            lengths   = batch_d["lengths"]
            B, T, _ = dx_logits.shape

            for b in range(B):
                L = int(lengths[b].item())
                for t in range(min(L - 1, T - 1)):
                    label = int(dx_seq[b, t + 1].item())
                    if label < 0:
                        continue
                    all_true.append(label)
                    all_pred.append(int(dx_logits[b, t].argmax().item()))

    if not all_true:
        return 0.0
    return float(accuracy_score(all_true, all_pred))


# ─── 真实图像增益消融（按源步有无图拆分）──────────────────────────────────
# 旧 evaluate_ablation 在全体样本上算 acc，被 ~59% 无图样本稀释，ΔIMG 看不出来。
# 这里只统计"源步 t 实际有图像"的预测样本，对比 full vs tab_only，
# 得到图像对预测的真实增益 ΔIMG。这是替代"贡献率 proxy"的真实优化目标。

def evaluate_image_gain(model,
                        loader,
                        device: str) -> Dict:
    """
    对验证集做真实图像增益消融，按源步是否有图像拆分统计准确率与 mAUC。

    返回 dict:
      acc_full_img / mauc_full_img : full 配置下、源步有图样本指标
      acc_tab_img / mauc_tab_img   : tab_only 配置下、同一批有图样本指标
      delta_img / delta_img_mauc   : full - tab_only（图像真实增益，关键指标）
      acc_full_noimg               : full 配置下、源步无图样本 acc
      n_img / n_noimg              : 两类样本数
    """
    def _run(no_img: bool):
        model.eval()
        t_img, p_img, prob_img = [], [], []
        t_no, p_no = [], []
        with torch.no_grad():
            for batch in loader:
                bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                      for k, v in batch.items()}
                orig_mod = bd["mod_avail"].clone()   # 用原始可用性判定"是否有图"
                if no_img:
                    bd.pop("mri_seq", None)
                    bd.pop("pet_seq", None)
                    ma = bd["mod_avail"].clone()
                    ma[:, :, 0] = 0
                    ma[:, :, 1] = 0
                    bd["mod_avail"] = ma
                out = model(bd, is_training=False)
                logits = out["dx_preds"]
                dx = bd["dx_seq"]
                lengths = bd["lengths"]
                B, T, _ = logits.shape
                for b in range(B):
                    L = int(lengths[b].item())
                    for t in range(min(L - 1, T - 1)):
                        lab = int(dx[b, t + 1].item())
                        if lab < 0:
                            continue
                        prob = F.softmax(logits[b, t], dim=0).cpu().numpy()
                        pred = int(prob.argmax())
                        has = (orig_mod[b, t, 0] > 0) or (orig_mod[b, t, 1] > 0)
                        if has:
                            t_img.append(lab); p_img.append(pred); prob_img.append(prob)
                        else:
                            t_no.append(lab); p_no.append(pred)
        return t_img, p_img, prob_img, t_no, p_no

    ti_f, pi_f, probi_f, tn_f, pn_f = _run(no_img=False)   # full
    ti_t, pi_t, probi_t, tn_t, pn_t = _run(no_img=True)    # tab_only

    def _acc(yt, yp):
        return float(accuracy_score(yt, yp)) if yt else float("nan")

    def _mauc(yt, probs):
        if not yt:
            return float("nan")
        return compute_classification_metrics(
            np.array(yt),
            np.array(probs).argmax(axis=1),
            np.array(probs),
        ).get("mauc", float("nan"))

    acc_full_img = _acc(ti_f, pi_f)
    acc_tab_img = _acc(ti_t, pi_t)
    mauc_full_img = _mauc(ti_f, probi_f)
    mauc_tab_img = _mauc(ti_t, probi_t)

    return {
        "acc_full_img"  : acc_full_img,
        "acc_tab_img"   : acc_tab_img,
        "delta_img"     : acc_full_img - acc_tab_img if ti_f and ti_t else float("nan"),
        "mauc_full_img" : mauc_full_img,
        "mauc_tab_img"  : mauc_tab_img,
        "delta_img_mauc": mauc_full_img - mauc_tab_img
                          if not math.isnan(mauc_full_img) and not math.isnan(mauc_tab_img)
                          else float("nan"),
        "acc_full_noimg": _acc(tn_f, pn_f),
        "n_img"         : len(ti_f),
        "n_noimg"       : len(tn_f),
    }


# ─── 打印指标 ─────────────────────────────────────────────────────────────────

def print_metrics(metrics: Dict) -> None:
    lines = []
    if "acc"  in metrics: lines.append(f"  Acc   : {metrics['acc']:.4f}")
    if "pre"  in metrics: lines.append(f"  Pre   : {metrics['pre']:.4f}")
    if "rec"  in metrics: lines.append(f"  Rec   : {metrics['rec']:.4f}")
    if "mauc" in metrics: lines.append(f"  mAUC  : {metrics['mauc']:.4f}")
    if "mae"  in metrics: lines.append(f"  MAE   : {metrics['mae']:.4f}")
    if "mre"  in metrics: lines.append(f"  MRE   : {metrics['mre']:.4f}")
    print("\n".join(lines))


# ─── 独立评估脚本 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from models.lmdp_net import LMDPNet
    from dataset import build_dataloaders

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="模型权重路径（.pth 文件）")
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--no_images",  action="store_true")
    parser.add_argument("--device",     default=Config.DEVICE)
    args = parser.parse_args()

    load_images = not args.no_images
    _, val_loader = build_dataloaders(
        fold_idx=args.fold,
        load_images=load_images,
    )

    # latent_dim 始终用 Config.LATENT_DIM：无图像时 M3VAE 自动返回零向量，
    # 不能传 0——否则与训练时的 checkpoint 维度不匹配导致 load_state_dict 报错。
    model = LMDPNet(
        latent_dim    = Config.LATENT_DIM,
        hidden_dim    = Config.HIDDEN_DIM,
        non_img_dim   = Config.NON_IMG_DIM,
        num_classes   = Config.NUM_CLASSES,
        biomarker_dim = Config.BIOMARKER_DIM,
    ).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    print(f"加载权重: {args.checkpoint}（epoch {ckpt.get('epoch', '?')}）")

    metrics = evaluate_fold(model, val_loader, args.device, load_images)
    print_metrics(metrics)
