import os
import torch

class Config:
    # ─── Paths ────────────────────────────────────────────────────────────────
    DATA_DIR            = "data"
    ADNIMERGE_PATH      = os.path.join(DATA_DIR, "ADNIMERGE.csv")
    MRI_RAW_DIR         = os.path.join(DATA_DIR, "mri_raw")       # raw NIfTI MRI files
    PET_RAW_DIR         = os.path.join(DATA_DIR, "pet_raw")       # raw NIfTI PET files
    MRI_PREP_DIR        = os.path.join(DATA_DIR, "mri_preprocessed")
    PET_PREP_DIR        = os.path.join(DATA_DIR, "pet_preprocessed")
    TAB_PROCESSED_PATH  = os.path.join(DATA_DIR, "tabular_processed.pkl")
    MNI152_TEMPLATE     = os.path.join(DATA_DIR, "MNI152_T1_1mm.nii.gz")

    # ─── Visit schedule ───────────────────────────────────────────────────────
    # 3年随访：bl, m06, m12, m24, m36（0-3年），用于研究3年内的进展预测
    VISIT_CODES = ["bl", "m06", "m12", "m24", "m36"]
    VISIT_MONTHS = {"bl": 0, "m06": 6, "m12": 12, "m24": 24, "m36": 36}

    # ─── Diagnosis ────────────────────────────────────────────────────────────
    # Map ADNIMERGE DX column to integers
    DX_MAP = {
        "CN": 0,
        "EMCI": 1, "MCI": 1, "LMCI": 1,
        "Dementia": 2, "AD": 2,
    }
    NUM_CLASSES = 3   # CN=0, MCI=1, AD=2

    # ─── Feature columns ──────────────────────────────────────────────────────
    BIOMARKER_COLS = ["Ventricles", "Hippocampus", "WholeBrain",
                      "Entorhinal", "Fusiform", "MidTemp"]
    ICV_COL        = "ICV"
    BIOMARKER_DIM  = 6

    # demographics: age(1) + education(1) + gender_one_hot(2) = 4
    DEMOGRAPHIC_DIM = 4
    # genetics: APOE4 one-hot {0,1,2} → 3
    GENETIC_DIM = 3

    NON_IMG_DIM = BIOMARKER_DIM + DEMOGRAPHIC_DIM + GENETIC_DIM  # 13

    # Mask covers 2 imaging availability flags + 13 non-imaging = 15
    MASK_DIM = 2 + NON_IMG_DIM   # 15

    # ─── Image preprocessing ──────────────────────────────────────────────────
    # 配准后的图像与 MNI152 模板同尺寸（依模板而定，不再假设固定 182×218×182）。
    # 网络输入固定为 128×160×128。如何把脑放进这个框由 PREP_FIT_MODE 决定。
    TARGET_SHAPE = (128, 160, 128)

    # PREP_FIT_MODE：配准后如何得到 128×160×128
    #   "resample_fit" （默认，推荐）：
    #       先按模板脑掩膜定位脑包围盒，重采样使整脑（含 margin）完整落入
    #       128×160×128。1mm 下整脑约 145×181×155mm 装不下，此模式会
    #       自动降到约 1.3–1.4mm，保证“脑完整 + 四周留背景”，根治过裁。
    #   "center_1mm" （贴论文）：
    #       保持 1mm，按模板脑中心做居中裁剪。颅顶/额枕极/外侧颞叶最边缘
    #       会被对称切掉（中央 AD 结构完整），与原论文 182×218×182→128×160×128
    #       的做法等价。
    PREP_FIT_MODE = "center_1mm"

    # resample_fit / center_1mm 共用：脑包围盒检测阈值（占模板最大强度的比例）
    PREP_BRAIN_THR_FRAC = 0.10
    # resample_fit：脑包围盒外保留的边界余量（体素，作用于目标网格）
    PREP_FIT_MARGIN = 6

    # 归一化：按分位数裁剪后再 min-max，避免极少数超亮体素压低整脑信号。
    # （旧实现用全局 min/max，对离群亮点敏感——见直方图 max=1 但 p99≈0.66）
    PREP_NORM_PCT = (0.5, 99.5)

    # ─ center_1mm 模式的固定裁剪窗口（仅当模板为标准 182×218×182 时成立）─
    # 注：当前 data/MNI152_T1_1mm.nii.gz 实为 ICBM152-2009（197×233×189），
    # 与下方窗口不匹配。center_1mm 模式会改为按模板脑中心动态居中，
    # 不再依赖这对静态偏移。保留仅作参考。
    ORIGINAL_SHAPE = (182, 218, 182)
    CROP_START = (27, 29, 27)
    CROP_END   = (155, 189, 155)

    # ─── Model hyperparameters ────────────────────────────────────────────────
    LATENT_DIM  = 256   # M3VAE latent dimension
    HIDDEN_DIM  = 256   # LSTM hidden state size (128→256：显存充裕，提升表达能力)
    IMG_CNN_CHANNELS = [32, 64, 128, 256, 256]  # 3D CNN channels

    # LSTM input = fused_mu (LATENT_DIM) + non_img (NON_IMG_DIM)
    LSTM_INPUT_DIM = LATENT_DIM + NON_IMG_DIM   # 269

    # ─── Training ─────────────────────────────────────────────────────────────
    LEARNING_RATE    = 0.001
    LR_FACTOR        = 0.5    # ReduceLROnPlateau: 验证 mAUC 停滞时学习率减半
    LR_PATIENCE      = 3      # 连续 3 个 epoch 无 mAUC 提升就降 LR
    MIN_LR           = 1e-5   # 学习率下限，避免后期震荡
    EARLY_STOP_PATIENCE = 12  # 连续 12 个 epoch 无改善则早停
    EARLY_STOP_MIN_DELTA = 1e-4
    IMG_GAIN_WEIGHT = 0.25     # 模型选择分数 = mAUC + 0.25 × 有图样本 ΔmAUC
    BATCH_SIZE       = 4      # 2→4，显存约 2.6→5GB（8GB 总量仍有余量）
    GRAD_ACCUM_STEPS = 2      # 有效 batch = BATCH_SIZE × GRAD_ACCUM_STEPS = 8
    USE_CHECKPOINT   = True   # 3D CNN 梯度检查点（以计算换显存）
    IMAGE_CACHE_SIZE = 64     # 每个 DataLoader worker 缓存的 .npy 图像数量；0=关闭
    ABLATION_INTERVAL = 10    # 每隔多少 epoch 做一次消融/图像增益验证；0=关闭
    NUM_EPOCHS       = 100
    K_FOLDS          = 5
    SEED             = 42
    MAX_SEQ_LEN      = 5    # bl, m06, m12, m24, m36（3年随访）

    # ─── 类别加权（应对 progressive 比例下降）─────────────────────────────────
    # 3年随访下 progressive 比例从 22.4% 降到 18.4%，类别失衡加剧。
    # 权重设计：progressive 类（MCI→AD, CN→MCI, CN→AD）权重更高，
    # 让模型认真学习进展样本，避免"全猜 stable"。
    # 这里按诊断类别加权：CN=0, MCI=1, AD=2
    # 逆频率加权：weight[c] = N_total / (NUM_CLASSES * N_c)
    # 训练集统计（3年随访, filter_mri_subjects）：
    #   CN: ~3300, MCI: ~3500, AD: ~1500（近似）
    # 计算后约 CN=0.88, MCI=0.83, AD=1.93，放大 AD 权重因 AD 样本少
    # 但 progressive 不等于 AD，需要进一步放大 MCI→AD 转化样本的权重。
    # 简化方案：直接给 AD 标签更高权重（因为 progressive 多指向 AD）
    DX_CLASS_WEIGHTS = [1.0, 1.2, 2.0]   # CN, MCI, AD
    DX_MASK_PROB     = 0.2  # 诊断标签不是模型输入；只轻微 mask，避免主任务/图像辅助监督过弱

    # KL weight in VAE loss (β-VAE).
    # At init, KL term (~24.7) is ~500x larger than recon (~0.05),
    # causing posterior collapse. β=0.001 rebalances so reconstruction dominates.
    # KL_WEIGHT     = 0.0003  # 降低 KL 压力，减少 posterior collapse 驱动力
    # KL_WEIGHT = 0.001
    # 2026-06-25: 训练日志显示 KL=177 是 recon=0.018 的 1万倍，KL×β=0.177 仍是
    # recon 的 10 倍梯度，编码器 mu 被持续拉向 0，fused_mu≈0 退化为噪声，
    # 消融实验 ΔMRI<0（有图反而更差）。降 10 倍让重建梯度占主导。
    KL_WEIGHT = 0.0001

    # ─── 图像编码器目标：降低辅助头主导性，优先服务纵向主任务 ───────────────
    # 最新训练日志显示：IMG_AUX_WEIGHT=3.0 时 laux*3 占总 loss 的大头，
    # 训练 loss 持续下降但验证 mAUC/图像增益下降，说明辅助判别目标过强且泛化不稳。
    # 因此把辅助监督改为弱正则：保留少量 future 诊断监督，让图像 latent 对齐
    # h_t→dx[t+1] 主任务；关闭 current 诊断监督，避免只学当前状态/扫描噪声。

    # Reconstruction loss weight. 保留弱自监督稳定项，避免纯辅助头过拟合。
    # 2026-06-25: recon=0.018 远小于 lp=0.8，编码器几乎感受不到重建梯度，
    # 提到 3.0 强迫图像信息必须保留在 fused_mu 中。
    RECON_WEIGHT  = 2.0

    # 图像辅助判别损失总权重。加强以强迫图像 latent 学到诊断相关特征。
    IMG_AUX_WEIGHT = 2.0
    IMG_AUX_CURRENT_WEIGHT = 1.0
    IMG_AUX_FUTURE_WEIGHT = 1.0

    # ─── 图像主导融合：图像作为 IRLSTM 主干，表格只作为残差信息 ─────────────
    # L_p 仍是主诊断目标；L_i 降权，避免插补/表格目标压过图像判别目标。
    LP_WEIGHT = 1.0
    LI_WEIGHT = 0.5
    LF_WEIGHT = 1.0

    # 表格残差强度：u_t = image_projector(fused_mu) + gate * scale * tab_projector(x)
    # 0.25 让表格能提供上下文，但不能像直接 concat 那样主导隐藏态。
    TAB_RESIDUAL_SCALE = 0.25

    # 训练时随机丢弃部分生物标志物值，削弱表格捷径；不改变 mask/标签。
    TAB_DROPOUT_PROB = 0.3

    # 贡献率下限惩罚权重（原硬编码在 m3vae.py 中为 0.5）。
    # 这是直接优化 proxy（精度权重 ≥40%）的项，与真实目标 ΔIMG 脱钩，关闭。
    CONTRIB_FLOOR_WEIGHT = 0.0

    WEIGHT_DECAY  = 5e-4   # L2 regularization in Adam
    DROPOUT       = 0.4    # dropout before prediction heads

    # ─── Hardware ─────────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
