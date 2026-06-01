"""
M3VAE（Multimodal Neuroimaging Representation Fusion）

论文 Section III.B：
  - 为每个模态（MRI, PET）设计独立的 3D CNN 编码器
  - 各编码器将输入映射到高斯分布参数 (μ, σ)
  - 使用 Product of Experts (PoE) 融合多模态分布
  - 多个解码器重建各模态输入
  - 训练时枚举所有 2^C-1 个模态子集（C=2 时共 3 种）
  - 推理时使用实际可用模态的融合 μ 作为下游特征
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# ─── 3D CNN 编码器 ────────────────────────────────────────────────────────────

class ImageEncoder3D(nn.Module):
    """
    3D CNN 编码器，将 128×160×128 图像映射到 (μ, log_σ)。

    Conv 链（stride=2）：
      1×128×160×128
      → 32×64×80×64
      → 64×32×40×32
      → 128×16×20×16
      → 256×8×10×8
      → 256×4×5×4
    Flatten → 20480 → Linear → (μ, log_σ) ∈ R^{latent_dim}
    """

    def __init__(self, latent_dim: int = Config.LATENT_DIM):
        super().__init__()
        chs = Config.IMG_CNN_CHANNELS  # [32, 64, 128, 256, 256]
        self.encoder = nn.Sequential(
            nn.Conv3d(1,      chs[0], 3, stride=2, padding=1), nn.BatchNorm3d(chs[0]), nn.LeakyReLU(0.2),
            nn.Conv3d(chs[0], chs[1], 3, stride=2, padding=1), nn.BatchNorm3d(chs[1]), nn.LeakyReLU(0.2),
            nn.Conv3d(chs[1], chs[2], 3, stride=2, padding=1), nn.BatchNorm3d(chs[2]), nn.LeakyReLU(0.2),
            nn.Conv3d(chs[2], chs[3], 3, stride=2, padding=1), nn.BatchNorm3d(chs[3]), nn.LeakyReLU(0.2),
            nn.Conv3d(chs[3], chs[4], 3, stride=2, padding=1), nn.BatchNorm3d(chs[4]), nn.LeakyReLU(0.2),
        )
        # After 5× stride-2: 4×5×4 spatial, chs[4] channels
        self.bottleneck_size = chs[4] * 4 * 5 * 4  # 20480
        self.fc = nn.Sequential(
            nn.Linear(self.bottleneck_size, 1024),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, 1, 128, 160, 128)
        返回: (μ, log_σ²)，各 shape = (B, latent_dim)
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-4.0, 2.0)
        return mu, logvar


# ─── 3D CNN 解码器 ────────────────────────────────────────────────────────────

class ImageDecoder3D(nn.Module):
    """
    3D CNN 解码器，将 latent_dim 向量重建为 128×160×128 图像。
    ConvTranspose3D(kernel=4, stride=2, padding=1) 每层将空间尺寸翻倍。
    """

    def __init__(self, latent_dim: int = Config.LATENT_DIM):
        super().__init__()
        chs = Config.IMG_CNN_CHANNELS[::-1]  # [256, 256, 128, 64, 32]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, chs[0] * 4 * 5 * 4),
            nn.ReLU(),
        )
        self.init_shape = (chs[0], 4, 5, 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(chs[0], chs[1], 4, stride=2, padding=1), nn.BatchNorm3d(chs[1]), nn.ReLU(),
            nn.ConvTranspose3d(chs[1], chs[2], 4, stride=2, padding=1), nn.BatchNorm3d(chs[2]), nn.ReLU(),
            nn.ConvTranspose3d(chs[2], chs[3], 4, stride=2, padding=1), nn.BatchNorm3d(chs[3]), nn.ReLU(),
            nn.ConvTranspose3d(chs[3], chs[4], 4, stride=2, padding=1), nn.BatchNorm3d(chs[4]), nn.ReLU(),
            nn.ConvTranspose3d(chs[4], 1,      4, stride=2, padding=1),
            nn.Sigmoid(),   # 输出值在 [0,1]（对应 min-max 归一化的图像）
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, latent_dim)
        返回: (B, 1, 128, 160, 128)
        """
        h = self.fc(z)
        h = h.view(h.size(0), *self.init_shape)
        return self.decoder(h)


# ─── Product of Experts 融合 ──────────────────────────────────────────────────

def product_of_experts(mu_list: list,
                        logvar_list: list,
                        mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    论文公式 (5)(6)(7)：使用 PoE 融合多个高斯分布。

    参数
    ----
    mu_list     : List of (B, latent_dim) 各模态的均值
    logvar_list : List of (B, latent_dim) 各模态的 log(σ²)
    mask        : (B, C) 二值掩码，1=该模态可用

    返回
    ----
    fused_mu     : (B, latent_dim)
    fused_logvar : (B, latent_dim)
    """
    # 精度 T_c = 1/σ²_c = exp(-logvar_c)  (对应论文 eq.5)
    B, latent_dim = mu_list[0].shape
    C = len(mu_list)

    # Accumulate precision and precision-weighted mean
    T_sum  = torch.zeros(B, latent_dim, device=mu_list[0].device)   # Σ T_c * m_c
    Tmu_sum = torch.zeros(B, latent_dim, device=mu_list[0].device)  # Σ μ_c * T_c * m_c

    # 始终加入标准正态先验 (μ=0, σ=1, T=1) 以确保分布有界
    T_sum   = T_sum   + 1.0
    Tmu_sum = Tmu_sum + 0.0

    for c in range(C):
        # mask[:, c]: (B,) → (B, 1) 广播
        m_c = mask[:, c].unsqueeze(1)                      # (B, 1)
        T_c = torch.exp(-logvar_list[c])                   # 精度 (B, latent_dim)
        T_sum   = T_sum   + T_c * m_c
        Tmu_sum = Tmu_sum + mu_list[c] * T_c * m_c

    # fused σ² = 1 / T_sum
    fused_var    = 1.0 / T_sum.clamp(min=1e-8)
    fused_logvar = torch.log(fused_var.clamp(min=1e-8))   # log(σ²_fused)
    fused_mu     = Tmu_sum * fused_var                    # (B, latent_dim)

    return fused_mu, fused_logvar


# ─── M3VAE 主体 ───────────────────────────────────────────────────────────────

class M3VAE(nn.Module):
    """
    Multimodal Neuroimaging Representation Fusion VAE。

    论文 Section III.B，涵盖：
      - 独立的 MRI / PET 编码器
      - PoE 多模态融合
      - 独立的 MRI / PET 解码器
      - 枚举所有模态子集进行训练

    使用方式
    --------
    forward(mri, pet, mri_avail, pet_avail):
      - 训练时：返回 loss（含重建 + KL 三种子集组合）
      - mri_avail / pet_avail：(B,) bool 张量，指示该批次中哪些样本有对应模态

    get_fused_mu(mri, pet, mri_avail, pet_avail):
      - 推理时：返回融合均值 (B, latent_dim)，用于下游 LSTM
    """

    def __init__(self, latent_dim: int = Config.LATENT_DIM):
        super().__init__()
        self.latent_dim  = latent_dim
        self.mri_encoder = ImageEncoder3D(latent_dim)
        self.pet_encoder = ImageEncoder3D(latent_dim)
        self.mri_decoder = ImageDecoder3D(latent_dim)
        self.pet_decoder = ImageDecoder3D(latent_dim)

    # ── 单模态编码（若该模态不可用，返回零均值/零方差）────────────────────
    def _encode_modality(self,
                          img: Optional[torch.Tensor],
                          encoder: nn.Module,
                          avail: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        img   : (B, 1, D, H, W) 或 None
        avail : (B,) bool
        返回 : mu, logvar，不可用样本置 0
        """
        B = avail.shape[0]
        mu     = torch.zeros(B, self.latent_dim, device=avail.device)
        logvar = torch.zeros(B, self.latent_dim, device=avail.device)

        idx = avail.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > 0 and img is not None:
            mu_sub, logvar_sub = encoder(img[idx])
            mu[idx]     = mu_sub.to(mu.dtype)
            logvar[idx] = logvar_sub.to(logvar.dtype)
        return mu, logvar

    # ── 重参数化采样 ──────────────────────────────────────────────────────────
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = μ + σ * ε，ε ~ N(0, I)（论文 eq.8）"""
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    # ── KL 散度（论文 eq.30）─────────────────────────────────────────────────
    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(N(μ,σ) || N(0,I)) = 0.5 * Σ(μ² + σ² - logσ² - 1)"""
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=-1)

    # ── 模态贡献率（基于 PoE 精度权重，仅用于日志）────────────────────────
    @staticmethod
    def compute_modality_contributions(
        mri_avail: torch.Tensor,
        pet_avail: torch.Tensor,
        mri_logvar: torch.Tensor,
        pet_logvar: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """
        返回 (contrib_mri, contrib_pet, contrib_prior)，三者之和约为 1。
        计算方式：T_c = exp(-logvar_c)，贡献率 = T_c * avail / T_total。
        先验精度固定为 1（标准正态）。
        不参与梯度计算。
        """
        T_mri   = torch.exp(-mri_logvar) * mri_avail.float().unsqueeze(1)  # (B, D)
        T_pet   = torch.exp(-pet_logvar) * pet_avail.float().unsqueeze(1)  # (B, D)
        T_total = 1.0 + T_mri + T_pet                                      # (B, D)

        c_mri   = (T_mri   / T_total).mean().detach().item()
        c_pet   = (T_pet   / T_total).mean().detach().item()
        c_prior = (1.0     / T_total).mean().detach().item()
        return c_mri, c_pet, c_prior

    # ── VAE 损失（论文 eq.29）────────────────────────────────────────────────
    def vae_loss(self,
                  mri: Optional[torch.Tensor],
                  pet: Optional[torch.Tensor],
                  mri_mu: torch.Tensor, mri_logvar: torch.Tensor,
                  pet_mu: torch.Tensor, pet_logvar: torch.Tensor,
                  mri_avail: torch.Tensor,
                  pet_avail: torch.Tensor) -> torch.Tensor:
        """
        对所有 2^C-1=3 种模态子集组合计算 VAE 损失并求和（论文 eq.29）。
        - 子集 {MRI}    : mask = [1, 0]
        - 子集 {PET}    : mask = [0, 1]
        - 子集 {MRI,PET}: mask = [1, 1]
        """
        B = mri_avail.shape[0]
        combo_losses = []
        recon_items  = []   # for logging only

        combos = [
            (torch.ones(B, dtype=torch.float32, device=mri_avail.device),
             torch.zeros(B, dtype=torch.float32, device=mri_avail.device)),   # {MRI}
            (torch.zeros(B, dtype=torch.float32, device=mri_avail.device),
             torch.ones(B, dtype=torch.float32, device=mri_avail.device)),    # {PET}
            (mri_avail.float(), pet_avail.float()),                            # {可用模态}
        ]

        # 各模态 encoder 的 KL 只算一次，避免被三个 combo 重复惩罚
        per_mod_kl_terms = []
        if mri_avail.any():
            per_mod_kl_terms.append(
                self.kl_divergence(mri_mu[mri_avail], mri_logvar[mri_avail]).mean())
        if pet_avail.any():
            per_mod_kl_terms.append(
                self.kl_divergence(pet_mu[pet_avail], pet_logvar[pet_avail]).mean())
        per_mod_kl = torch.stack(per_mod_kl_terms).mean() if per_mod_kl_terms else \
                     (mri_mu.sum() * 0.0)

        for mask_mri, mask_pet in combos:
            combo_avail = (mask_mri + mask_pet).clamp(max=1).bool()
            if not combo_avail.any():
                continue

            mask = torch.stack([mask_mri, mask_pet], dim=1)
            fused_mu, fused_logvar = product_of_experts(
                [mri_mu, pet_mu], [mri_logvar, pet_logvar], mask)
            z = self.reparameterize(fused_mu, fused_logvar)

            recon_losses = []
            if mri_avail.any() and mri is not None:
                idx_m = mri_avail.nonzero(as_tuple=False).squeeze(1)
                recon_mri = self.mri_decoder(z[idx_m])
                recon_losses.append(F.mse_loss(recon_mri, mri[idx_m]))
                del recon_mri
            if pet_avail.any() and pet is not None:
                idx_p = pet_avail.nonzero(as_tuple=False).squeeze(1)
                recon_pet = self.pet_decoder(z[idx_p])
                recon_losses.append(F.mse_loss(recon_pet, pet[idx_p]))
                del recon_pet

            if recon_losses:
                recon_loss = torch.stack(recon_losses).mean()
            else:
                recon_loss = next(self.mri_decoder.parameters()).sum() * 0.0

            combo_losses.append(recon_loss + Config.KL_WEIGHT * per_mod_kl)
            recon_items.append(recon_loss.detach().item())

        avg_recon = sum(recon_items) / len(recon_items) if recon_items else 0.0
        avg_kl    = per_mod_kl.detach().item()

        if combo_losses:
            return torch.stack(combo_losses).mean(), avg_recon, avg_kl
        return (mri_mu.sum() + pet_mu.sum()) * 0.0, 0.0, 0.0

    # ── 获取推理用融合均值（论文 eq.10）──────────────────────────────────────
    @torch.no_grad()
    def get_fused_mu(self,
                      mri: Optional[torch.Tensor],
                      pet: Optional[torch.Tensor],
                      mri_avail: torch.Tensor,
                      pet_avail: torch.Tensor) -> torch.Tensor:
        """
        推理时使用融合均值 μ_fused（不采样随机噪声），避免随机性影响下游任务。
        返回: (B, latent_dim)
        """
        mri_mu, mri_logvar = self._encode_modality(mri, self.mri_encoder, mri_avail)
        pet_mu, pet_logvar = self._encode_modality(pet, self.pet_encoder, pet_avail)

        mask = torch.stack([mri_avail.float(), pet_avail.float()], dim=1)
        fused_mu, _ = product_of_experts(
            [mri_mu, pet_mu],
            [mri_logvar, pet_logvar],
            mask,
        )

        # 若两个模态都不可用，返回零向量
        no_img = (~mri_avail & ~pet_avail)
        if no_img.any():
            fused_mu[no_img] = 0.0

        return fused_mu

    # ── 训练时的完整前向传播 ─────────────────────────────────────────────────
    def forward(self,
                mri: Optional[torch.Tensor],
                pet: Optional[torch.Tensor],
                mri_avail: torch.Tensor,
                pet_avail: torch.Tensor):
        """
        训练时使用。
        返回 (fused_mu, vae_loss, contribs, avg_recon, avg_kl)
          fused_mu  : (B, latent_dim)
          vae_loss  : 标量（recon + β·KL）
          contribs  : (c_mri, c_pet, c_prior) 贡献率日志
          avg_recon : float  平均重建损失（监控用）
          avg_kl    : float  平均原始 KL（监控用，未乘 β）
        """
        mri_mu, mri_logvar = self._encode_modality(mri, self.mri_encoder, mri_avail)
        pet_mu, pet_logvar = self._encode_modality(pet, self.pet_encoder, pet_avail)

        loss, avg_recon, avg_kl = self.vae_loss(
            mri, pet, mri_mu, mri_logvar, pet_mu, pet_logvar, mri_avail, pet_avail)

        mask = torch.stack([mri_avail.float(), pet_avail.float()], dim=1)
        fused_mu, _ = product_of_experts(
            [mri_mu, pet_mu], [mri_logvar, pet_logvar], mask)

        contribs = self.compute_modality_contributions(
            mri_avail, pet_avail, mri_logvar, pet_logvar)

        return fused_mu, loss, contribs, avg_recon, avg_kl
