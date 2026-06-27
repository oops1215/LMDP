"""
LMDP-Net（Longitudinal Multimodal Disease Progression Network）
	完整模型：M3VAE + 插补模块 + IRLSTM + 预测头
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
    MinimalRNN 风格的插补模块，仅预测生物标志物（6D）以填充缺失值。

    论文 Section III 描述：
      用前一时间步的预测值填充缺失值。
      imputed_bio = bio_mask * bio_t + (1 - bio_mask) * bio_pred

    具体实现：
      bio_pred = W_imp * h_{t-1} + b_imp（从前一步隐藏态预测 6 维生物标志物）
      非生物标志物特征（年龄、教育、性别、APOE4）始终用实测值，不做插补。
    """

    def __init__(self,
                 hidden_dim:    int = Config.HIDDEN_DIM,
                 non_img_dim:   int = Config.NON_IMG_DIM,
                 biomarker_dim: int = Config.BIOMARKER_DIM):
        super().__init__()
        self.biomarker_dim = biomarker_dim
        self.non_img_dim   = non_img_dim
        # 只预测生物标志物维度（年龄/教育/遗传学不需要插补）
        self.predictor = nn.Linear(hidden_dim, biomarker_dim)

    def forward(self,
                x_t:       torch.Tensor,
                bio_mask_t: torch.Tensor,
                h_prev:    torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_t        : (B, non_img_dim)   当前访视全特征（含缺失生物标志物，缺失处为 0）
        bio_mask_t : (B, biomarker_dim) 生物标志物观测掩码，1=实测，0=缺失
        h_prev     : (B, hidden_dim)    上一步隐藏状态（第一步时为全零）

        返回
        ----
        x_imputed : (B, non_img_dim)   插补后的全特征（非生物标志物维度直接用实测值）
        bio_pred  : (B, biomarker_dim) 生物标志物预测值（用于 L_i 损失）
        """
        bio_pred    = self.predictor(h_prev)                             # (B, biomarker_dim)
        bio_imputed = bio_mask_t * x_t[:, :self.biomarker_dim] + \
                      (1.0 - bio_mask_t) * bio_pred                     # (B, biomarker_dim)
        # 非生物标志物维度（人口统计+遗传）始终使用实测值
        x_imputed = torch.cat([bio_imputed, x_t[:, self.biomarker_dim:]], dim=1)
        return x_imputed, bio_pred


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
                 latent_dim:    int = Config.LATENT_DIM,
                 hidden_dim:    int = Config.HIDDEN_DIM,
                 non_img_dim:   int = Config.NON_IMG_DIM,
                 num_classes:   int = Config.NUM_CLASSES,
                 biomarker_dim: int = Config.BIOMARKER_DIM,
                 use_checkpoint: bool = Config.USE_CHECKPOINT):
        super().__init__()
        self.latent_dim    = latent_dim
        self.hidden_dim    = hidden_dim
        self.non_img_dim   = non_img_dim
        self.num_classes   = num_classes
        self.biomarker_dim = biomarker_dim

        # 类别加权交叉熵权重（应对 progressive 比例下降导致的类别失衡）
        # CN=0, MCI=1, AD=2；AD 权重更高，因为 progressive 多指向 AD
        dx_weights = getattr(Config, "DX_CLASS_WEIGHTS", None)
        if dx_weights is not None:
            self.dx_class_weights = torch.tensor(dx_weights, dtype=torch.float32)
        else:
            self.dx_class_weights = None

        # 模块
        self.m3vae      = M3VAE(latent_dim, use_checkpoint=use_checkpoint)
        self.imputation = ImputationModule(hidden_dim, non_img_dim, biomarker_dim)

        # 图像主导输入：先把 fused_mu 投到 hidden_dim，表格只作为受限残差。
        self.image_projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.tab_projector = nn.Sequential(
            nn.Linear(non_img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.tab_gate = nn.Sequential(
            nn.Linear(latent_dim + non_img_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.irlstm     = IRLSTM(
            input_dim  = hidden_dim,
            hidden_dim = hidden_dim,
            mask_dim   = Config.MASK_DIM,            # 15
        )

        # 预测头（论文 eq.25, eq.26）+ dropout 防止过拟合
        self.dropout  = nn.Dropout(p=Config.DROPOUT)
        self.pred_dx  = nn.Linear(hidden_dim, num_classes)
        self.pred_bio = nn.Linear(hidden_dim, biomarker_dim)

        # 辅助图像判别头：强迫 fused_mu 学习与诊断相关的特征
        # 直接从图像潜在向量预测诊断，权重 IMG_AUX_WEIGHT 控制强度
        self.img_aux_head = nn.Linear(latent_dim, num_classes)

    # ── 辅助：从批次中提取某时间步的图像 ──────────────────────────────────────
    @staticmethod
    def _extract_step(tensor_or_none: Optional[torch.Tensor],
                      t: int) -> Optional[torch.Tensor]:
        if tensor_or_none is None:
            return None
        return tensor_or_none[:, t]   # (B, 1, D, H, W)

    def _apply_tabular_dropout(self,
                               non_img_t: torch.Tensor,
                               bio_mask_t: torch.Tensor) -> torch.Tensor:
        if not self.training or Config.TAB_DROPOUT_PROB <= 0:
            return non_img_t
        bio = non_img_t[:, :self.biomarker_dim]
        rest = non_img_t[:, self.biomarker_dim:]
        keep = (torch.rand_like(bio) > Config.TAB_DROPOUT_PROB).float()
        bio = bio * (keep + (1.0 - bio_mask_t))
        return torch.cat([bio, rest], dim=1)

    def _build_lstm_input(self,
                          fused_mu: torch.Tensor,
                          x_imputed: torch.Tensor,
                          mod_avail_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 无图时 fused_mu 是从 prior 采样的随机噪声，直接投影会污染隐藏态。
        # 用 has_img 掩码把无图时的图像分支置零，让模型明确区分有图/无图状态。
        has_img = (mod_avail_t.sum(dim=1, keepdim=True) > 0).float()
        img = self.image_projector(fused_mu) * has_img
        tab = self.tab_projector(x_imputed)
        gate_in = torch.cat([fused_mu * has_img, x_imputed, mod_avail_t], dim=1)
        gate = self.tab_gate(gate_in)
        tab_scale = has_img * Config.TAB_RESIDUAL_SCALE + (1.0 - has_img)
        tab_residual = gate * tab_scale * tab
        u_t = img + tab_residual
        explain = {
            "gate": gate,
            "gate_mean": gate.mean(),
            "image_component": img,
            "tabular_component": tab,
            "tabular_residual": tab_residual,
            "tabular_scale": tab_scale,
        }
        return u_t, explain

    # ── 主前向传播 ───────────────────────────────────────────────────────────
    def forward(self,
                batch: Dict,
                is_training: bool = True,
                return_explain: bool = False,
                enable_cam: bool = False):
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
        all_fused_mu  = []    # 收集各步 fused_mu，用于辅助图像分类损失
        all_img_avail = []    # 对应步骤是否有图像（有图才算辅助损失）
        gate_logs     = []    # 表格残差门控强度

        h = torch.zeros(B, self.hidden_dim, device=device)
        c = torch.zeros(B, self.hidden_dim, device=device)
        hiddens = []
        explain_steps = []

        for t in range(T):
            mri_t   = self._extract_step(mri_seq, t)    # (B,1,D,H,W) or None
            pet_t   = self._extract_step(pet_seq, t)
            mri_av  = mod_avail[:, t, 0].bool()         # (B,)
            pet_av  = mod_avail[:, t, 1].bool()         # (B,)

            # ── M3VAE ──────────────────────────────────────────────────────
            m3_explain = None
            if is_training:
                if return_explain:
                    fused_mu, vae_loss, contribs, recon, kl, m3_explain = self.m3vae(
                        mri_t, pet_t, mri_av, pet_av, return_explain=True)
                else:
                    fused_mu, vae_loss, contribs, recon, kl = self.m3vae(
                        mri_t, pet_t, mri_av, pet_av)
                vae_losses.append(vae_loss)
                all_contribs.append(contribs)
                recon_logs.append(recon)
                kl_logs.append(kl)
                all_fused_mu.append(fused_mu)
                all_img_avail.append((mri_av | pet_av))   # 该步骤是否有任意图像
            else:
                if return_explain:
                    m3_explain = self.m3vae.encode_for_explain(
                        mri_t, pet_t, mri_av, pet_av, enable_cam=enable_cam)
                    fused_mu = m3_explain["fused_mu"]
                else:
                    fused_mu = self.m3vae.get_fused_mu(mri_t, pet_t, mri_av, pet_av)

            # ── 插补（使用上一步 LSTM 隐藏态）──────────────────────────────
            non_img_t  = non_img_seq[:, t, :]        # (B, non_img_dim)
            bio_mask_t = bio_mask_seq[:, t, :]       # (B, 6)
            non_img_for_model = self._apply_tabular_dropout(non_img_t, bio_mask_t)

            # 传入 bio_mask_t（6D），ImputationModule 内部只对生物标志物插补
            x_imputed, bio_pred = self.imputation(non_img_for_model, bio_mask_t, h)
            all_x_pred.append(bio_pred)              # (B, biomarker_dim)

            # ── IRLSTM 单步：图像主干 + 受限表格残差 ───────────────────────
            u_t, step_lstm_explain = self._build_lstm_input(fused_mu, x_imputed, mod_avail[:, t, :])
            gate_logs.append(step_lstm_explain["gate_mean"])
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

            if return_explain:
                explain_steps.append({
                    "m3vae": m3_explain,
                    "fused_mu": fused_mu,
                    "x_imputed": x_imputed,
                    "bio_impute_pred": bio_pred,
                    "u_t": u_t,
                    "hidden": h,
                    "cell": c,
                    "mod_avail": mod_avail[:, t, :],
                    **step_lstm_explain,
                })

        # ── 拼接隐藏态序列 ───────────────────────────────────────────────────
        hidden_seq = torch.cat(hiddens, dim=1)  # (B, T, hidden_dim)

        # ── 预测头：预测下一时间步（论文 eq.25, eq.26）──────────────────────
        # 从第 t 步的隐藏态预测第 t+1 步的诊断和生物标志物
        h_drop         = self.dropout(hidden_seq)
        dx_logits_seq  = self.pred_dx(h_drop)        # (B, T, num_classes)
        bio_pred_seq   = self.pred_bio(h_drop)       # (B, T, biomarker_dim)

        if not is_training:
            out = {
                "dx_preds"  : dx_logits_seq,   # softmax 在外面做
                "bio_preds" : bio_pred_seq,
            }
            if return_explain:
                out["explain"] = self._pack_explain(explain_steps, hidden_seq, dx_logits_seq, bio_pred_seq)
            return out

        # ── 损失计算 ─────────────────────────────────────────────────────────

        # L_p：诊断预测损失（论文 eq.28）
        # 用第 t 步隐藏态预测第 t+1 到 t+K 的诊断
        lp = self._compute_lp(dx_logits_seq, dx_seq, lengths)

        # L_i：生物标志物插补损失（论文 eq.29 中的 L_i，MAE）
        li = self._compute_li(all_x_pred, non_img_seq, bio_mask_seq, lengths, T)

        # L_f：VAE 损失（论文 eq.29 中的 L_f）
        lf = torch.stack(vae_losses).mean() if vae_losses else torch.tensor(0.0, device=device)

        # L_aux：图像辅助判别损失
        # 同时预测当前诊断和下一步诊断；下一步诊断与主任务 L_p 对齐，
        # 避免 fused_mu 只学到当前状态但不改善纵向预测。
        l_aux = self._compute_img_aux(all_fused_mu, all_img_avail, dx_seq, lengths, T)

        # L_total：诊断 + 降权插补 + 图像 VAE + 图像辅助判别
        # 图像主导实验中，L_i 只保留为弱正则，避免表格插补目标压过图像判别目标。
        total = (Config.LP_WEIGHT * lp +
                 Config.LI_WEIGHT * li +
                 Config.LF_WEIGHT * lf +
                 Config.IMG_AUX_WEIGHT * l_aux)

        # 各模态平均贡献率（跨时间步平均，contribs 为 5 元组）
        if all_contribs:
            n = len(all_contribs)
            avg_c_mri      = sum(c[0] for c in all_contribs) / n
            avg_c_pet      = sum(c[1] for c in all_contribs) / n
            avg_c_prior    = sum(c[2] for c in all_contribs) / n
            avg_c_mri_cond = sum(c[3] for c in all_contribs) / n  # 有 MRI 时的条件贡献
            avg_c_pet_cond = sum(c[4] for c in all_contribs) / n  # 有 PET 时的条件贡献
        else:
            avg_c_mri = avg_c_pet = avg_c_mri_cond = avg_c_pet_cond = 0.0
            avg_c_prior = 1.0

        avg_recon = sum(recon_logs) / len(recon_logs) if recon_logs else 0.0
        avg_kl    = sum(kl_logs)    / len(kl_logs)    if kl_logs    else 0.0
        avg_gate  = torch.stack(gate_logs).mean().detach().item() if gate_logs else 0.0

        out = {
            "total_loss"       : total,
            "lp"               : lp.item(),
            "li"               : li.item(),
            "lf"               : lf.item(),
            "l_aux"            : l_aux.item(),
            "lf_recon"         : avg_recon,
            "lf_kl"            : avg_kl,
            "gate_mean"        : avg_gate,
            "dx_preds"         : dx_logits_seq,
            "bio_preds"        : bio_pred_seq,
            "contrib_mri"      : avg_c_mri,
            "contrib_pet"      : avg_c_pet,
            "contrib_prior"    : avg_c_prior,
            "contrib_mri_cond" : avg_c_mri_cond,  # 有 MRI 时的真实编码器贡献
            "contrib_pet_cond" : avg_c_pet_cond,  # 有 PET 时的真实编码器贡献
        }
        if return_explain:
            out["explain"] = self._pack_explain(explain_steps, hidden_seq, dx_logits_seq, bio_pred_seq)
        return out

    @staticmethod
    def _stack_step_values(explain_steps: List[Dict], key: str) -> Optional[torch.Tensor]:
        vals = [step.get(key) for step in explain_steps]
        if not vals or any(v is None for v in vals):
            return None
        return torch.stack(vals, dim=1)

    @staticmethod
    def _stack_m3_values(explain_steps: List[Dict], key: str) -> Optional[torch.Tensor]:
        vals = []
        for step in explain_steps:
            m3 = step.get("m3vae")
            if m3 is None or m3.get(key) is None:
                return None
            vals.append(m3[key])
        if not vals:
            return None
        return torch.stack(vals, dim=1)

    def _pack_explain(self,
                      explain_steps: List[Dict],
                      hidden_seq: torch.Tensor,
                      dx_logits_seq: torch.Tensor,
                      bio_pred_seq: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """把逐时间步解释信息整理成 (B,T,...) 张量，便于反事实/可视化脚本使用。"""
        explain = {
            "dx_logits_seq": dx_logits_seq,
            "bio_pred_seq": bio_pred_seq,
            "hidden_seq": hidden_seq,
            "fused_mu_seq": self._stack_step_values(explain_steps, "fused_mu"),
            "x_imputed_seq": self._stack_step_values(explain_steps, "x_imputed"),
            "bio_impute_pred_seq": self._stack_step_values(explain_steps, "bio_impute_pred"),
            "u_seq": self._stack_step_values(explain_steps, "u_t"),
            "tab_gate_seq": self._stack_step_values(explain_steps, "gate"),
            "image_component_seq": self._stack_step_values(explain_steps, "image_component"),
            "tabular_component_seq": self._stack_step_values(explain_steps, "tabular_component"),
            "tabular_residual_seq": self._stack_step_values(explain_steps, "tabular_residual"),
            "tabular_scale_seq": self._stack_step_values(explain_steps, "tabular_scale"),
            "mod_avail_seq": self._stack_step_values(explain_steps, "mod_avail"),
            "mri_mu_seq": self._stack_m3_values(explain_steps, "mri_mu"),
            "mri_logvar_seq": self._stack_m3_values(explain_steps, "mri_logvar"),
            "pet_mu_seq": self._stack_m3_values(explain_steps, "pet_mu"),
            "pet_logvar_seq": self._stack_m3_values(explain_steps, "pet_logvar"),
            "fused_logvar_seq": self._stack_m3_values(explain_steps, "fused_logvar"),
            "contrib_mri_seq": self._stack_m3_values(explain_steps, "contrib_mri_sample"),
            "contrib_pet_seq": self._stack_m3_values(explain_steps, "contrib_pet_sample"),
            "contrib_prior_seq": self._stack_m3_values(explain_steps, "contrib_prior_sample"),
            "contrib_mri_dim_seq": self._stack_m3_values(explain_steps, "contrib_mri"),
            "contrib_pet_dim_seq": self._stack_m3_values(explain_steps, "contrib_pet"),
            "contrib_prior_dim_seq": self._stack_m3_values(explain_steps, "contrib_prior"),
        }
        return explain

    # ── 诊断预测损失（多步）────────────────────────────────────────────────
    def _compute_lp(self,
                    logits: torch.Tensor,   # (B, T, num_classes)
                    labels: torch.Tensor,   # (B, T)  int
                    lengths: torch.Tensor   # (B,)
    ) -> torch.Tensor:
        """论文 eq.28：用 h_t 预测下一步 t+1 的诊断（单步）。"""
        B, T, C = logits.shape
        losses = []
        weights = self.dx_class_weights.to(logits.device) if self.dx_class_weights is not None else None

        for b in range(B):
            L = int(lengths[b].item())
            for t in range(min(L - 1, T - 1)):   # h_t → label[t+1]
                label = labels[b, t + 1].item()
                if label < 0:
                    continue
                losses.append(F.cross_entropy(
                    logits[b, t].unsqueeze(0),
                    torch.tensor([label], device=logits.device, dtype=torch.long),
                    weight=weights
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

        for t in range(1, T):   # t=0 时 h_{-1}=0，无历史信息，跳过
            x_pred_t = x_pred_list[t]                           # (B, biomarker_dim) 已是 6D
            x_true_t = non_img_seq[:, t, :self.biomarker_dim]   # (B, 6)
            mask_t   = bio_mask[:, t, :]                         # (B, 6)

            active = (mask_t > 0)
            if active.any():
                losses.append(F.l1_loss(x_pred_t[active], x_true_t[active]))

        if losses:
            return torch.stack(losses).mean()
        return non_img_seq.sum() * 0.0  # 全缺失时返回可微分的零

    # ── 图像辅助判别损失 ──────────────────────────────────────────────────────
    def _compute_img_aux(self,
                         fused_mu_list: list,    # T × (B, latent_dim)
                         img_avail_list: list,   # T × (B,) bool
                         dx_seq: torch.Tensor,   # (B, T)
                         lengths: torch.Tensor,
                         T: int) -> torch.Tensor:
        """
        对源步有图像的 latent 加弱诊断监督。
        默认只监督 future loss，使 fused_mu 对齐主任务 h_t→dx[t+1]；
        current loss 可通过 Config.IMG_AUX_CURRENT_WEIGHT 打开，但默认关闭，
        避免辅助头只拟合当前状态/扫描噪声而压过纵向预测目标。
        """
        zero = fused_mu_list[0].sum() * 0.0
        use_current = Config.IMG_AUX_CURRENT_WEIGHT > 0
        use_future = Config.IMG_AUX_FUTURE_WEIGHT > 0
        if not use_current and not use_future:
            return zero

        current_losses = []
        future_losses = []
        weights = self.dx_class_weights.to(fused_mu_list[0].device) if self.dx_class_weights is not None else None
        for t in range(T):
            fused_mu_t = fused_mu_list[t]          # (B, latent_dim)
            img_av_t   = img_avail_list[t]          # (B,) bool
            if not img_av_t.any():
                continue
            logits_t = self.img_aux_head(fused_mu_t)
            for b in range(fused_mu_t.shape[0]):
                if not img_av_t[b]:
                    continue
                L = int(lengths[b].item())
                if t >= L:
                    continue

                if use_current:
                    current_label = dx_seq[b, t].item()
                    if current_label >= 0:
                        current_losses.append(F.cross_entropy(
                            logits_t[b].unsqueeze(0),
                            torch.tensor([current_label], device=fused_mu_t.device, dtype=torch.long),
                            weight=weights
                        ))

                if use_future and t + 1 < L:
                    future_label = dx_seq[b, t + 1].item()
                    if future_label >= 0:
                        future_losses.append(F.cross_entropy(
                            logits_t[b].unsqueeze(0),
                            torch.tensor([future_label], device=fused_mu_t.device, dtype=torch.long),
                            weight=weights
                        ))

        current_loss = torch.stack(current_losses).mean() if current_losses else zero
        future_loss = torch.stack(future_losses).mean() if future_losses else zero
        return (Config.IMG_AUX_CURRENT_WEIGHT * current_loss +
                Config.IMG_AUX_FUTURE_WEIGHT * future_loss)


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
