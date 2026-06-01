"""
IRLSTM（Information-Regularized LSTM）

论文 Section III.C：
  核心改动（相对于标准 LSTM）：
  1. 隐藏状态时间衰减：γ_h = exp(-max(0, W_γh·δ + b_γh))，h̃ = h_{t-1} * γ_h
  2. Mask 向量加入门控计算（区分实测值/插补值）
  3. 辅助函数修正遗忘门（使其值更接近 0 或 1）：
       g = f - sin(f·π)·cos(f·π) / π    (论文 eq.22)
  4. 细胞状态更新使用 g 替代 f：
       c_t = g ⊙ c_{t-1} + (1-g) ⊙ c̃_t  (eq.23)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# ─── 辅助函数 ψ(f) 可视化/验证 ──────────────────────────────────────────────

def auxiliary_forget(f: torch.Tensor) -> torch.Tensor:
    """
    论文 eq.22：g = f - sin(f·π)·cos(f·π) / π
    等价于 g = f - sin(2·f·π) / (2π)（利用二倍角公式）
    满足：
      - 在 f=0.5 时 g=0.5（对称）
      - g 单调递增
      - g ∈ [0, 1]（当 β=π）
    """
    pi = math.pi
    g = f - torch.sin(f * pi) * torch.cos(f * pi) / pi
    return g.clamp(0.0, 1.0)


# ─── IRLSTM 单个时间步（Cell）────────────────────────────────────────────────

class IRLSTMCell(nn.Module):
    """
    IRLSTM 单步单元，实现论文 eq.(16)-(24)。

    参数
    ----
    input_dim  : 输入 u_t 的维度（= LATENT_DIM + NON_IMG_DIM）
    hidden_dim : 隐藏状态 h 的维度（论文最优值 = 128）
    mask_dim   : 掩码向量 m_t 的维度（= 2 + NON_IMG_DIM = 15）
    """

    def __init__(self,
                 input_dim:  int = Config.LSTM_INPUT_DIM,
                 hidden_dim: int = Config.HIDDEN_DIM,
                 mask_dim:   int = Config.MASK_DIM):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.mask_dim   = mask_dim

        # 隐藏状态时间衰减参数（论文 eq.16）
        self.W_gamma_h = nn.Linear(1, hidden_dim, bias=True)   # δ 是标量

        # 细胞候选门（c̃，对应 input gate in standard LSTM）(eq.18)
        self.W_uc = nn.Linear(input_dim,  hidden_dim, bias=False)
        self.W_hc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_mc = nn.Linear(mask_dim,   hidden_dim, bias=True)

        # 输出门 (eq.19)
        self.W_uo = nn.Linear(input_dim,  hidden_dim, bias=False)
        self.W_ho = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_mo = nn.Linear(mask_dim,   hidden_dim, bias=True)

        # 遗忘门 (eq.20)
        self.W_uf = nn.Linear(input_dim,  hidden_dim, bias=False)
        self.W_hf = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_mf = nn.Linear(mask_dim,   hidden_dim, bias=True)

    def forward(self,
                u_t:    torch.Tensor,
                m_t:    torch.Tensor,
                delta:  torch.Tensor,
                h_prev: torch.Tensor,
                c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        u_t    : (B, input_dim)  主输入（fused_mu + 非图像特征）
        m_t    : (B, mask_dim)   掩码向量（哪些特征是实测值）
        delta  : (B, 1)          距上次访视的时间间隔（月）
        h_prev : (B, hidden_dim)
        c_prev : (B, hidden_dim)

        返回
        ----
        h_t : (B, hidden_dim)
        c_t : (B, hidden_dim)
        """
        # ── 1. 隐藏状态时间衰减（eq.16, eq.17）──────────────────────────
        gamma_h = torch.exp(-F.softplus(self.W_gamma_h(delta)))   # (B, hidden_dim)
        h_est   = h_prev * gamma_h                                 # 时间衰减后的估计隐藏态

        # ── 2. 细胞候选（eq.18）────────────────────────────────────────
        c_tilde = torch.tanh(self.W_uc(u_t) + self.W_hc(h_est) + self.W_mc(m_t))

        # ── 3. 输出门（eq.19）──────────────────────────────────────────
        o_t = torch.sigmoid(self.W_uo(u_t) + self.W_ho(h_est) + self.W_mo(m_t))

        # ── 4. 遗忘门 + 辅助函数（eq.20, eq.22）───────────────────────
        f_t = torch.sigmoid(self.W_uf(u_t) + self.W_hf(h_est) + self.W_mf(m_t))
        g_t = auxiliary_forget(f_t)   # 修正遗忘门

        # ── 5. 细胞状态更新（eq.23）────────────────────────────────────
        c_t = g_t * c_prev + (1.0 - g_t) * c_tilde

        # ── 6. 隐藏状态（eq.24）────────────────────────────────────────
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


# ─── IRLSTM 序列模块 ─────────────────────────────────────────────────────────

class IRLSTM(nn.Module):
    """
    包装 IRLSTMCell，处理变长序列。

    forward 接受：
      inputs  : (B, T, input_dim)
      masks   : (B, T, mask_dim)
      deltas  : (B, T, 1)         每个时间步距上次访视的月数
      lengths : (B,)              每个样本的实际序列长度
    返回：
      hiddens : (B, T, hidden_dim)
    """

    def __init__(self,
                 input_dim:  int = Config.LSTM_INPUT_DIM,
                 hidden_dim: int = Config.HIDDEN_DIM,
                 mask_dim:   int = Config.MASK_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = IRLSTMCell(input_dim, hidden_dim, mask_dim)

    def forward(self,
                inputs:  torch.Tensor,
                masks:   torch.Tensor,
                deltas:  torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = inputs.shape
        device  = inputs.device

        h = torch.zeros(B, self.hidden_dim, device=device)
        c = torch.zeros(B, self.hidden_dim, device=device)

        hiddens = []
        for t in range(T):
            # 只更新序列未结束的样本，已结束的保持 h 不变
            active = (t < lengths).float().unsqueeze(1)  # (B, 1)

            h_new, c_new = self.cell(
                inputs[:, t, :],   # u_t
                masks[:, t, :],    # m_t
                deltas[:, t, :],   # δ_t
                h, c,
            )
            h = h_new * active + h * (1.0 - active)
            c = c_new * active + c * (1.0 - active)
            hiddens.append(h.unsqueeze(1))

        return torch.cat(hiddens, dim=1)  # (B, T, hidden_dim)
