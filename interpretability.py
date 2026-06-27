"""
反事实与可解释性工具。

该模块不参与训练主流程，主要用于论文实验中的：
  1. 模态级反事实：full / no-MRI / no-PET / tab-only 的概率差；
  2. biomarker 反事实：把指定非图像特征替换为参考值后的预测变化；
  3. latent 反事实：在 fused_mu 空间中寻找最小扰动以改变目标诊断；
  4. 3D Grad-CAM：基于 encoder conv5 feature map 的基础热图。
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from config import Config
from evaluate import evaluate_ablation, evaluate_image_gain


BIOMARKER_NAMES = Config.BIOMARKER_COLS


def _to_device(batch: Dict, device: str) -> Dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def clone_batch(batch: Dict) -> Dict:
    copied = {}
    for k, v in batch.items():
        copied[k] = v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
    return copied


def apply_modality_counterfactual(batch: Dict,
                                  no_mri: bool = False,
                                  no_pet: bool = False) -> Dict:
    """返回关闭指定影像模态后的 batch，不修改原 batch。"""
    cf = clone_batch(batch)
    if no_mri:
        cf.pop("mri_seq", None)
    if no_pet:
        cf.pop("pet_seq", None)
    if (no_mri or no_pet) and "mod_avail" in cf:
        ma = cf["mod_avail"].clone()
        if no_mri:
            ma[:, :, 0] = 0
        if no_pet:
            ma[:, :, 1] = 0
        cf["mod_avail"] = ma
    return cf


@torch.no_grad()
def predict_probabilities(model,
                          batch: Dict,
                          device: str,
                          return_explain: bool = False) -> Dict:
    """运行模型并返回 softmax 概率；默认不计算梯度。"""
    model.eval()
    bd = _to_device(batch, device)
    out = model(bd, is_training=False, return_explain=return_explain)
    probs = F.softmax(out["dx_preds"], dim=-1)
    out = dict(out)
    out["dx_probs"] = probs
    return out


@torch.no_grad()
def modality_counterfactual_report(model,
                                    batch: Dict,
                                    device: str,
                                    class_idx: int = 2) -> Dict[str, torch.Tensor]:
    """
    对同一 batch 计算 full/no_mri/no_pet/tab_only 的类别概率差。

    class_idx 默认 2，即 AD 类。
    返回张量 shape 通常为 (B,T)。
    """
    full = predict_probabilities(model, batch, device)["dx_probs"]
    no_mri = predict_probabilities(
        model, apply_modality_counterfactual(batch, no_mri=True), device)["dx_probs"]
    no_pet = predict_probabilities(
        model, apply_modality_counterfactual(batch, no_pet=True), device)["dx_probs"]
    tab_only = predict_probabilities(
        model, apply_modality_counterfactual(batch, no_mri=True, no_pet=True), device)["dx_probs"]

    p_full = full[..., class_idx]
    p_no_mri = no_mri[..., class_idx]
    p_no_pet = no_pet[..., class_idx]
    p_tab_only = tab_only[..., class_idx]

    return {
        "p_full": p_full,
        "p_no_mri": p_no_mri,
        "p_no_pet": p_no_pet,
        "p_tab_only": p_tab_only,
        "delta_mri": p_full - p_no_mri,
        "delta_pet": p_full - p_no_pet,
        "delta_img": p_full - p_tab_only,
    }


def dataset_modality_counterfactual_report(model,
                                           loader,
                                           device: str) -> Dict:
    """复用现有评估逻辑，输出验证集层面的模态反事实/消融指标。"""
    return {
        "acc_full": evaluate_ablation(model, loader, device),
        "acc_no_mri": evaluate_ablation(model, loader, device, no_mri=True),
        "acc_no_pet": evaluate_ablation(model, loader, device, no_pet=True),
        "acc_tab_only": evaluate_ablation(model, loader, device, no_mri=True, no_pet=True),
        "image_gain": evaluate_image_gain(model, loader, device),
    }


@torch.no_grad()
def biomarker_counterfactual_report(model,
                                    batch: Dict,
                                    device: str,
                                    reference_values: torch.Tensor,
                                    class_idx: int = 2,
                                    time_index: Optional[int] = None) -> Dict:
    """
    逐个替换 biomarker 为 reference_values，观察目标类别概率变化。

    reference_values: (BIOMARKER_DIM,) 或 (T, BIOMARKER_DIM)，需与 non_img_seq 同一标准化空间。
    time_index=None 表示替换所有时间步；否则只替换指定 visit。
    """
    base = predict_probabilities(model, batch, device)["dx_probs"][..., class_idx]
    bd = _to_device(batch, device)
    ref = reference_values.to(device)
    deltas = {}

    for j, name in enumerate(BIOMARKER_NAMES):
        cf = clone_batch(bd)
        if time_index is None:
            if ref.dim() == 1:
                cf["non_img_seq"][:, :, j] = ref[j]
            else:
                cf["non_img_seq"][:, :, j] = ref[:, j].unsqueeze(0)
        else:
            value = ref[j] if ref.dim() == 1 else ref[time_index, j]
            cf["non_img_seq"][:, time_index, j] = value
        prob_cf = predict_probabilities(model, cf, device)["dx_probs"][..., class_idx]
        deltas[name] = {
            "p_cf": prob_cf,
            "delta": base - prob_cf,
        }

    return {
        "p_base": base,
        "features": deltas,
    }


def collect_explanations(model,
                         loader,
                         device: str,
                         max_batches: Optional[int] = None) -> List[Dict]:
    """收集 return_explain=True 的输出，便于后续统计贡献率和 gate。"""
    model.eval()
    outputs = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            bd = _to_device(batch, device)
            out = model(bd, is_training=False, return_explain=True)
            outputs.append({
                "dx_probs": F.softmax(out["dx_preds"], dim=-1).cpu(),
                "explain": {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                            for k, v in out["explain"].items()},
                "lengths": bd["lengths"].cpu(),
                "dx_seq": bd["dx_seq"].cpu(),
            })
    return outputs


def summarize_modality_contributions(explanation_outputs: List[Dict]) -> Dict[str, float]:
    """汇总 collect_explanations 的 MRI/PET/prior 贡献率。"""
    vals = {"mri": [], "pet": [], "prior": [], "tab_gate": []}
    for out in explanation_outputs:
        exp = out["explain"]
        for src, key in [("mri", "contrib_mri_seq"),
                         ("pet", "contrib_pet_seq"),
                         ("prior", "contrib_prior_seq")]:
            v = exp.get(key)
            if isinstance(v, torch.Tensor):
                vals[src].append(v.reshape(-1))
        gate = exp.get("tab_gate_seq")
        if isinstance(gate, torch.Tensor):
            vals["tab_gate"].append(gate.reshape(-1))

    summary = {}
    for k, parts in vals.items():
        if parts:
            x = torch.cat(parts)
            summary[f"{k}_mean"] = float(x.mean().item())
            summary[f"{k}_std"] = float(x.std(unbiased=False).item())
    return summary


def latent_counterfactual(model,
                          batch: Dict,
                          device: str,
                          sample_index: int,
                          time_index: int,
                          target_class: int,
                          steps: int = 100,
                          lr: float = 1e-2,
                          l2_weight: float = 1e-2,
                          prior_weight: float = 1e-3) -> Dict[str, torch.Tensor]:
    """
    在指定样本/时间步的 fused_mu 上优化 delta，使预测转向 target_class。

    注意：该函数优化的是 LSTM 输入处的 latent，并固定其它时间步 latent/表格输入不变；
    适合做论文中的 latent-level counterfactual，不会修改模型参数。
    """
    model.eval()
    bd = _to_device(batch, device)
    with torch.no_grad():
        out = model(bd, is_training=False, return_explain=True)
        exp = out["explain"]
        z_seq = exp["fused_mu_seq"].detach().clone()
        x_imp_seq = exp["x_imputed_seq"].detach().clone()
        base_logits = out["dx_preds"][sample_index, time_index].detach()
        base_prob = F.softmax(base_logits, dim=0)

    delta = torch.zeros_like(z_seq[sample_index, time_index], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    target = torch.tensor([target_class], device=device, dtype=torch.long)

    for _ in range(steps):
        opt.zero_grad()
        z_cf = z_seq.clone()
        z_cf[sample_index, time_index] = z_cf[sample_index, time_index] + delta

        h = torch.zeros(z_seq.shape[0], model.hidden_dim, device=device)
        c = torch.zeros_like(h)
        hiddens = []
        for t in range(z_seq.shape[1]):
            u_t, _ = model._build_lstm_input(z_cf[:, t], x_imp_seq[:, t], bd["mod_avail"][:, t, :])
            m_t = torch.cat([
                bd["mod_avail"][:, t, :],
                bd["bio_mask_seq"][:, t, :],
                torch.ones(z_seq.shape[0], model.non_img_dim - model.biomarker_dim, device=device),
            ], dim=1)
            active = (t < bd["lengths"]).float().unsqueeze(1)
            h_new, c_new = model.irlstm.cell(u_t, m_t, bd["delta_seq"][:, t, :], h, c)
            h = h_new * active + h * (1.0 - active)
            c = c_new * active + c * (1.0 - active)
            hiddens.append(h.unsqueeze(1))
        hidden_cf = torch.cat(hiddens, dim=1)
        logits_cf = model.pred_dx(hidden_cf)[sample_index, time_index].unsqueeze(0)
        loss_cls = F.cross_entropy(logits_cf, target)
        loss_l2 = delta.pow(2).mean()
        loss_prior = z_cf[sample_index, time_index].pow(2).mean()
        loss = loss_cls + l2_weight * loss_l2 + prior_weight * loss_prior
        loss.backward()
        opt.step()

    with torch.no_grad():
        z_cf_single = z_seq[sample_index, time_index] + delta
        # 可选图像反事实：用现有 decoder 将 latent 解码回 MRI/PET 空间。
        mri_cf = model.m3vae.mri_decoder(z_cf_single.unsqueeze(0))
        pet_cf = model.m3vae.pet_decoder(z_cf_single.unsqueeze(0))

    return {
        "z_base": z_seq[sample_index, time_index].detach().cpu(),
        "z_cf": z_cf_single.detach().cpu(),
        "delta": delta.detach().cpu(),
        "base_prob": base_prob.detach().cpu(),
        "target_class": torch.tensor(target_class),
        "mri_cf": mri_cf.detach().cpu(),
        "pet_cf": pet_cf.detach().cpu(),
    }


def compute_3d_gradcam(model,
                       batch: Dict,
                       device: str,
                       modality: str = "mri",
                       sample_index: int = 0,
                       time_index: int = 0,
                       class_idx: int = 2) -> torch.Tensor:
    """基础 3D Grad-CAM，返回归一化到 [0,1] 的低分辨率 CAM。"""
    model.eval()
    bd = _to_device(batch, device)
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    out = model(bd, is_training=False, return_explain=True, enable_cam=True)
    score = out["dx_preds"][sample_index, time_index, class_idx]
    score.backward()

    encoder = model.m3vae.mri_encoder if modality.lower() == "mri" else model.m3vae.pet_encoder
    features = encoder.get_cam_features()
    gradients = encoder.get_cam_gradients()
    if features is None or gradients is None:
        raise RuntimeError(f"No Grad-CAM cache for modality={modality}; check modality availability.")

    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
    cam = (weights * features).sum(dim=1)
    cam = F.relu(cam)
    cam_sample = cam[min(sample_index, cam.shape[0] - 1)]
    cam_sample = cam_sample - cam_sample.min()
    cam_sample = cam_sample / cam_sample.max().clamp(min=1e-8)
    return cam_sample.detach().cpu()
