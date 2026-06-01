"""
LMDP-Net（Longitudinal Multimodal Disease Progression Network）

论文 Section III 完整模型，包含四个模块：
  1. M3VAE   : 多模态神经影像融合（处理 MRI/PET 模态缺失）
  2. Imputation : 非图像特征（生物标志物等）插补
  3. IRLSTM  : 改进遗忘门的时序编码
  4. Prediction : 诊断分类 + 生物标志物预测

输入约定（每次访视）：
  - mri          : (B, 1, 128, 160, 128) 或 None
  - pet          : (B, 1, 128, 160, 128) 或 None
  - non_img      : (B, NON_IMG_DIM)  [bio(6) + demo(4) + gen(3)]
  - bio_mask     : (B, 6)   生物标志物是否可观测
  - mod_avail    : (B, 2)   [mri_avail, pet_avail]
  - delta        : (B, 1)   距上次访视月数

输出（每个时间步）：
  - dx_pred      : (B, NUM_CLASSES) 诊断概率
  - bio_pred     : (B, BIOMARKER_DIM) 生物标志物预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from models.m3vae import M3VAE
from models.irlstm import IRLSTM


# ─── 插补模块（Imputation）────────────────────────────────────────────────────

class ImputationModule(nn.Module):
    """
    MinimalRNN 风格的插补模块，用于填充缺失的非图像特征（生物标志物）。

    论文 Section III 描述：
      用前一时间步的预测值填充缺失值。
      imputed_x = m * x + (1 - m) * x_pred

    具体实现：
      x_pred_t = W_imp * h_{t-1} + b_imp（从前一步隐藏态预测）
      imputed_x_t = m_t * x_t + (1 - m_t) * x_pred_t
    """

    def __init__(self,
                 hidden_dim:   int = Config.HIDDEN_DIM,
                 non_img_dim:  int = Config.NON_IMG_DIM):
        super().__init__()
        self.predictor = nn.Linear(hidden_dim, non_img_dim)

    def forward(self,
                x_t:    torch.Tensor,
                m_t:    torch.Tensor,
                h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_t    : (B, non_img_dim)  当前访视特征（含缺失值，缺失处为 0）
        m_t    : (B, non_img_dim)  掩码，1=实测，0=缺失
        h_prev : (B, hidden_dim)   上一步隐藏状态（第一步时为全零）

        返回
        ----
        x_imputed : (B, non_img_dim)  插补后的特征
        x_pred    : (B, non_img_dim)  模型对 x_t 的预测（用于 L_i 损失）
        """
        x_pred    = self.predictor(h_prev)                    # 从上一步隐藏态预测
        x_imputed = m_t * x_t + (1.0 - m_t) * x_pred        # 实测用实测，缺失用预测
        return x_imputed, x_pred


# ─── LMDP-Net 完整模型 ────────────────────────────────────────────────────────

class LMDPNet(nn.Module):
    """
    LMDP-Net 完整实现。

    forward(batch) 接受一个字典 batch，包含：
      mri_seq     : (B, T, 1, 128, 160, 128) or None（序列图像）
      pet_seq     : (B, T, 1, 128, 160, 128) or None
      non_img_seq : (B, T, NON_IMG_DIM)
      bio_mask_seq: (B, T, 6)       生物标志物观测掩码
      mod_avail   : (B, T, 2)       图像模态可用性
      delta_seq   : (B, T, 1)       时间间隔
      dx_seq      : (B, T)          诊断标签（-1 表示缺失）
      lengths     : (B,)            实际序列长度
      is_training : bool            是否计算完整损失

    返回（训练时）：{'total_loss', 'lp', 'li', 'lf', 'dx_preds', 'bio_preds'}
    返回（推理时）：{'dx_preds', 'bio_preds'}  按时间步预测
    """

    def __init__(self,
                 latent_dim:   int = Config.LATENT_DIM,
                 hidden_dim:   int = Config.HIDDEN_DIM,
                 non_img_dim:  int = Config.NON_IMG_DIM,
                 num_classes:  int = Config.NUM_CLASSES,
                 biomarker_dim: int = Config.BIOMARKER_DIM):
        super().__init__()
        self.latent_dim    = latent_dim
        self.hidden_dim    = hidden_dim
        self.non_img_dim   = non_img_dim
        self.num_classes   = num_classes
        self.biomarker_dim = biomarker_dim

        # 模块
        self.m3vae      = M3VAE(latent_dim)
        self.imputation = ImputationModule(hidden_dim, non_img_dim)
        self.irlstm     = IRLSTM(
            input_dim  = latent_dim + non_img_dim,  # 269
            hidden_dim = hidden_dim,
            mask_dim   = Config.MASK_DIM,            # 15
        )

        # 预测头（论文 eq.25, eq.26）+ dropout 防止过拟合
        self.dropout  = nn.Dropout(p=Config.DROPOUT)
        self.pred_dx  = nn.Linear(hidden_dim, num_classes)
        self.pred_bio = nn.Linear(hidden_dim, biomarker_dim)

    # ── 辅助：从批次中提取某时间步的图像 ──────────────────────────────────────
    @staticmethod
    def _extract_step(tensor_or_none: Optional[torch.Tensor],
                      t: int) -> Optional[torch.Tensor]:
        if tensor_or_none is None:
            return None
        return tensor_or_none[:, t]   # (B, 1, D, H, W)

    # ── 主前向传播 ───────────────────────────────────────────────────────────
    def forward(self, batch: Dict, is_training: bool = True):
        B     = batch["non_img_seq"].shape[0]
        T     = batch["non_img_seq"].shape[1]
        device = batch["non_img_seq"].device

        mri_seq      = batch.get("mri_seq")       # (B,T,1,D,H,W) or None
        pet_seq      = batch.get("pet_seq")       # (B,T,1,D,H,W) or None
        non_img_seq  = batch["non_img_seq"]        # (B,T,non_img_dim)
        bio_mask_seq = batch["bio_mask_seq"]       # (B,T,6)
        mod_avail    = batch["mod_avail"]          # (B,T,2)  float
        delta_seq    = batch["delta_seq"]          # (B,T,1)
        dx_seq       = batch["dx_seq"]             # (B,T)  int
        lengths      = batch["lengths"]            # (B,)   int

        all_x_pred    = []
        vae_losses    = []
        all_contribs  = []
        recon_logs    = []    # avg recon per step (logging)
        kl_logs       = []    # avg raw KL per step (logging)

        h = torch.zeros(B, self.hidden_dim, device=device)
        c = torch.zeros(B, self.hidden_dim, device=device)
        hiddens = []

        for t in range(T):
            mri_t   = self._extract_step(mri_seq, t)    # (B,1,D,H,W) or None
            pet_t   = self._extract_step(pet_seq, t)
            mri_av  = mod_avail[:, t, 0].bool()         # (B,)
            pet_av  = mod_avail[:, t, 1].bool()         # (B,)

            # ── M3VAE ──────────────────────────────────────────────────────
            if is_training:
                fused_mu, vae_loss, contribs, recon, kl = self.m3vae(
                    mri_t, pet_t, mri_av, pet_av)
                vae_losses.append(vae_loss)
                all_contribs.append(contribs)
                recon_logs.append(recon)
                kl_logs.append(kl)
            else:
                fused_mu = self.m3vae.get_fused_mu(mri_t, pet_t, mri_av, pet_av)

            # ── 插补（使用上一步 LSTM 隐藏态）──────────────────────────────
            non_img_t  = non_img_seq[:, t, :]        # (B, non_img_dim)
            bio_mask_t = bio_mask_seq[:, t, :]       # (B, 6)
            non_img_mask = torch.cat([
                bio_mask_t,
                torch.ones(B, self.non_img_dim - self.biomarker_dim, device=device)
            ], dim=1)  # (B, non_img_dim)

            x_imputed, x_pred = self.imputation(non_img_t, non_img_mask, h)
            all_x_pred.append(x_pred)

            # ── IRLSTM 单步（交替推进，使下一步插补能用到当前隐藏态）──────
            u_t = torch.cat([fused_mu, x_imputed], dim=1)  # (B, latent+non_img)
            m_t = torch.cat([
                mod_avail[:, t, :],                                                    # (B,2)
                bio_mask_t,                                                             # (B,6)
                torch.ones(B, self.non_img_dim - self.biomarker_dim, device=device),  # (B,7)
            ], dim=1)  # (B, mask_dim=15)

            active = (t < lengths).float().unsqueeze(1)   # (B,1)
            h_new, c_new = self.irlstm.cell(u_t, m_t, delta_seq[:, t, :], h, c)
            h = h_new * active + h * (1.0 - active)
            c = c_new * active + c * (1.0 - active)
            hiddens.append(h.unsqueeze(1))

        # ── 拼接隐藏态序列 ───────────────────────────────────────────────────
        hidden_seq = torch.cat(hiddens, dim=1)  # (B, T, hidden_dim)

        # ── 预测头：预测下一时间步（论文 eq.25, eq.26）──────────────────────
        # 从第 t 步的隐藏态预测第 t+1 步的诊断和生物标志物
        h_drop         = self.dropout(hidden_seq)
        dx_logits_seq  = self.pred_dx(h_drop)        # (B, T, num_classes)
        bio_pred_seq   = self.pred_bio(h_drop)       # (B, T, biomarker_dim)

        if not is_training:
            return {
                "dx_preds"  : dx_logits_seq,   # softmax 在外面做
                "bio_preds" : bio_pred_seq,
            }

        # ── 损失计算 ─────────────────────────────────────────────────────────

        # L_p：诊断预测损失（论文 eq.28）
        # 用第 t 步隐藏态预测第 t+1 到 t+K 的诊断
        lp = self._compute_lp(dx_logits_seq, dx_seq, lengths)

        # L_i：生物标志物插补损失（论文 eq.29 中的 L_i，MAE）
        li = self._compute_li(all_x_pred, non_img_seq, bio_mask_seq, lengths, T)

        # L_f：VAE 损失（论文 eq.29 中的 L_f）
        lf = torch.stack(vae_losses).mean() if vae_losses else torch.tensor(0.0, device=device)

        # L_total = L_p + L_i + L_f（论文 eq.27，等权重）
        total = lp + li + lf

        # 各模态平均贡献率（跨时间步平均）
        if all_contribs:
            n = len(all_contribs)
            avg_c_mri   = sum(c[0] for c in all_contribs) / n
            avg_c_pet   = sum(c[1] for c in all_contribs) / n
            avg_c_prior = sum(c[2] for c in all_contribs) / n
        else:
            avg_c_mri = avg_c_pet = 0.0
            avg_c_prior = 1.0

        avg_recon = sum(recon_logs) / len(recon_logs) if recon_logs else 0.0
        avg_kl    = sum(kl_logs)    / len(kl_logs)    if kl_logs    else 0.0

        return {
            "total_loss"   : total,
            "lp"           : lp.item(),
            "li"           : li.item(),
            "lf"           : lf.item(),
            "lf_recon"     : avg_recon,      # 重建损失（期望下降）
            "lf_kl"        : avg_kl,         # 原始 KL（未乘 β，供监控）
            "dx_preds"     : dx_logits_seq,
            "bio_preds"    : bio_pred_seq,
            "contrib_mri"  : avg_c_mri,
            "contrib_pet"  : avg_c_pet,
            "contrib_prior": avg_c_prior,
        }

    # ── 诊断预测损失（多步）────────────────────────────────────────────────
    def _compute_lp(self,
                    logits: torch.Tensor,   # (B, T, num_classes)
                    labels: torch.Tensor,   # (B, T)  int
                    lengths: torch.Tensor   # (B,)
    ) -> torch.Tensor:
        """论文 eq.28：用 h_t 预测下一步 t+1 的诊断（单步）。"""
        B, T, C = logits.shape
        losses = []

        for b in range(B):
            L = int(lengths[b].item())
            for t in range(min(L - 1, T - 1)):   # h_t → label[t+1]
                label = labels[b, t + 1].item()
                if label < 0:
                    continue
                losses.append(F.cross_entropy(
                    logits[b, t].unsqueeze(0),
                    torch.tensor([label], device=logits.device, dtype=torch.long)
                ))

        if losses:
            return torch.stack(losses).mean()
        return logits.sum() * 0.0

    # ── 插补损失 ──────────────────────────────────────────────────────────────
    def _compute_li(self,
                    x_pred_list: list,      # T × (B, non_img_dim)
                    non_img_seq: torch.Tensor,  # (B, T, non_img_dim)
                    bio_mask:    torch.Tensor,  # (B, T, 6)
                    lengths:     torch.Tensor,
                    T:           int
    ) -> torch.Tensor:
        """
        论文 eq.29：只对实测生物标志物值计算 MAE 插补损失。
        x_pred[t] 是对第 t 步生物标志物的预测（来自 h_{t-1}）。
        """
        losses = []

        for t in range(T):
            x_pred_t = x_pred_list[t][:, :self.biomarker_dim]  # (B, 6)
            x_true_t = non_img_seq[:, t, :self.biomarker_dim]   # (B, 6)
            mask_t   = bio_mask[:, t, :]                         # (B, 6)

            active = (mask_t > 0)
            if active.any():
                losses.append(F.l1_loss(x_pred_t[active], x_true_t[active]))

        if losses:
            return torch.stack(losses).mean()
        return non_img_seq.sum() * 0.0  # 全缺失时返回可微分的零


# ─── 简单单步推理接口 ────────────────────────────────────────────────────────

class LMDPNetInference(nn.Module):
    """
    推理时逐步调用 LMDP-Net 的辅助接口。
    适用于测试集评估或在线推理。
    """

    def __init__(self, model: LMDPNet):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def predict_sequence(self, batch: Dict) -> Dict:
        """
        对完整序列做推理，返回每步的诊断预测和生物标志物预测。
        """
        self.model.eval()
        out = self.model(batch, is_training=False)
        dx_probs = F.softmax(out["dx_preds"], dim=-1)   # (B, T, num_classes)
        return {
            "dx_probs"  : dx_probs,                     # 诊断概率
            "dx_class"  : dx_probs.argmax(dim=-1),      # 预测类别
            "bio_preds" : out["bio_preds"],              # 生物标志物预测
        }
