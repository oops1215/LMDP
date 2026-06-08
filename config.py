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
    # Paper uses M0, M12, M24, M36, M48, M60
    VISIT_CODES = ["bl", "m06", "m12", "m18", "m24", "m30", "m36", "m42", "m48", "m54", "m60"]
    VISIT_MONTHS = {"bl": 0, "m06": 6, "m12": 12, "m18": 18, "m24": 24, "m30": 30, "m36": 36, "m42": 42, "m48": 48, "m54": 54, "m60": 60}

    # ─── Diagnosis ────────────────────────────────────────────────────────────
    # Map ADNIMERGE DX column to integers
    DX_MAP = {
        "CN": 0, "SMC": 0,
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
    # After registration to MNI152 the image is 182×218×182
    # Crop to remove background → 128×160×128
    ORIGINAL_SHAPE = (182, 218, 182)
    TARGET_SHAPE   = (128, 160, 128)
    # Crop offsets: (182-128)//2=27, (218-160)//2=29, (182-128)//2=27
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
    BATCH_SIZE       = 4      # 2→4，显存约 2.6→5GB（8GB 总量仍有余量）
    GRAD_ACCUM_STEPS = 2      # 有效 batch = BATCH_SIZE × GRAD_ACCUM_STEPS = 8
    USE_CHECKPOINT   = True   # 3D CNN 梯度检查点（以计算换显存）
    NUM_EPOCHS       = 100
    K_FOLDS          = 5
    SEED             = 42
    MAX_SEQ_LEN      = 11  # max 11 visits (M0–M60 including intermediate)
    DX_MASK_PROB     = 0.5  # probability of masking current-visit diagnosis during training

    # KL weight in VAE loss (β-VAE).
    # At init, KL term (~24.7) is ~500x larger than recon (~0.05),
    # causing posterior collapse. β=0.001 rebalances so reconstruction dominates.
    KL_WEIGHT     = 0.001

    # Reconstruction loss weight. Default rec≈0.009 << lp≈0.8, encoder barely
    # feels reconstruction gradient. RECON_WEIGHT=50 brings rec to ~0.45,
    # forcing encoder to actually retain image information in z.
    RECON_WEIGHT  = 50.0

    WEIGHT_DECAY  = 5e-4
    DROPOUT       = 0.4

    # ─── Hardware ─────────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
