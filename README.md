# LMDP-Net 复现指南

**论文**：Dao et al., "Longitudinal Alzheimer's Disease Progression Prediction With Modality Uncertainty and Optimization of Information Flow", IEEE J. Biomed. Health Inform., 2025.

> 本指南按**从零开始**的顺序写成，跟着每一步执行即可。

---

## 目录

1. [环境准备](#1-环境准备)
2. [申请 ADNI 数据权限](#2-申请-adni-数据权限)
3. [下载表格数据（ADNIMERGE2）](#3-下载表格数据adnimerge2)
4. [生成 MRI 下载清单](#4-生成-mri-下载清单)
5. [在 ADNI IDA 下载 MRI 图像](#5-在-adni-ida-下载-mri-图像)
6. [在 ADNI IDA 下载 PET 图像](#6-在-adni-ida-下载-pet-图像)
7. [整理图像文件](#7-整理图像文件)
8. [预处理表格数据](#8-预处理表格数据)
9. [预处理图像数据](#9-预处理图像数据)
10. [训练模型](#10-训练模型)
11. [评估模型](#11-评估模型)

---

## 项目结构

```
LMDP/
├── config.py                           # 所有超参数与路径配置
├── run_pipeline.py                     # 端到端一键运行脚本
├── dataset.py                          # PyTorch Dataset / DataLoader
├── train.py                            # 训练（5-fold CV）
├── evaluate.py                         # 评估指标
├── requirements.txt
├── data/
│   ├── build_adnimerge_from_rda.py     # 从 .rda 文件构建 ADNIMERGE.csv
│   ├── generate_mri_download_list.py   # 从 UCSFFSX*.rda 提取 MRI Image ID 清单
│   ├── download_adni.py                # 图像文件整理工具
│   ├── preprocess_tabular.py           # 表格数据预处理
│   └── preprocess_imaging.py           # MRI/PET 图像配准+裁剪（替代 SPM12）
└── models/
    ├── m3vae.py                        # M³VAE（多模态融合，含 PoE）
    ├── irlstm.py                       # IRLSTM（改进遗忘门的 LSTM）
    └── lmdp_net.py                     # LMDP-Net 完整模型
```

---

## 1. 环境准备

```bash
# 克隆代码
git clone https://github.com/oops1215/LMDP.git
cd LMDP

# 安装依赖
pip install -r requirements.txt
```

> **注意**：`antspyx` 安装需要几分钟（C++ 编译）。若失败：
> ```bash
> pip install antspyx --no-build-isolation
> ```

**硬件要求**：
- 图像预处理：≥ 32 GB RAM
- 训练（多模态）：≥ 16 GB GPU 显存（推荐 A6000 / V100）
- 训练（仅表格）：普通 GPU 即可

---

## 2. 申请 ADNI 数据权限

1. 访问 [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)
2. 点击右上角 **Apply for Access** → 填写研究目的（学术研究/论文复现）
3. 审核通常需要 **1–2 个工作日**，通过后会收到邮件
4. 同一账号可同时访问 ADNI IDA（图像库）和 LONI（表格数据）

---

## 3. 下载表格数据（ADNIMERGE2）

### 3-A  下载 ADNIMERGE2 R 包

登录 [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)：

```
Study Data → Study Info → Data & Databases
→ "ADNIMERGE" → 下载 ADNIMERGE2.tar.gz
```

### 3-B  解压并构建 ADNIMERGE.csv

```bash
# 解压到本地（示例路径，可自定义）
tar -xzf ADNIMERGE2.tar.gz -C ~/

# 安装 pyreadr（读取 .rda 文件）
pip install pyreadr

# 先诊断列名（可选，确认 .rda 文件内容正常）
python data/build_adnimerge_from_rda.py \
    --rda_dir ~/ADNIMERGE2/data \
    --inspect_only

# 生成 data/ADNIMERGE.csv
python data/build_adnimerge_from_rda.py \
    --rda_dir ~/ADNIMERGE2/data \
    --output  data/ADNIMERGE.csv
```

**预期输出**：`data/ADNIMERGE.csv`，包含以下关键列：

| 列名 | 含义 |
|------|------|
| PTID | 受试者 ID（`XXX_S_XXXX`）|
| VISCODE | 访视代码（`bl` / `m12` / `m24` / `m36` / `m48` / `m60`）|
| DX | 诊断（`CN` / `MCI` / `Dementia`）|
| AGE / PTGENDER / PTEDUCAT / APOE4 | 人口统计学 |
| Ventricles / Hippocampus / WholeBrain / Entorhinal / Fusiform / MidTemp / ICV | FreeSurfer 脑区体积 |

> **替代方案**：若已有 ADNIMERGE.csv，直接放到 `data/ADNIMERGE.csv` 跳过本步。

---

## 4. 生成 MRI 下载清单

FreeSurfer 脑区体积（UCSFFSX*.rda）是在特定 MRI 扫描上计算的，由 **IMAGEUID**（= ADNI IDA 中的 Image Data ID）唯一标识。用此脚本提取精确的 Image ID：

```bash
python data/generate_mri_download_list.py \
    --rda_dir ~/ADNIMERGE2/data \
    --output  data/mri_download_list.csv
```

**输出两个文件**：
- `data/mri_download_list.csv`：含 PTID、VISCODE、IMAGEUID、EXAMDATE 等
- `data/mri_download_list_ids_only.txt`：纯数字 Image ID 列表，用于 ADNI IDA 搜索框

---

## 5. 在 ADNI IDA 下载 MRI 图像

登录 [https://ida.loni.usc.edu/](https://ida.loni.usc.edu/)

### 方法 A：按 Image Data ID 精确搜索（推荐）

1. **Download → Image Collections → Advanced Image Search**
2. 在 **Image ID** 栏粘贴 `data/mri_download_list_ids_only.txt` 中的 ID（支持多行粘贴）
3. 点击 **Search**
4. **Select All** → **Add to Collection**（命名如 `MRI_LMDP`）
5. 进入 My Collections → 选中集合 → **Advanced Download**
6. Format 选 **NIfTI** → 下载压缩包和 **CSV 清单**（后续整理用）

> 此方法下载的图像与 FreeSurfer 体积一一对应，无歧义。

### 方法 B：按 Visit + Description 手动筛选

若无法使用方法 A，按以下条件搜索（每个 ADNI 阶段分别搜索）：

**Modality = MRI，Description = `MPR; GradWarp; B1 Correction; N3; Scaled`**

**Visit 勾选对照表**：

| 论文访视 | 需要勾选的 Visit 选项 |
|---------|---------------------|
| bl | ADNI Baseline、ADNI2 Baseline-New Pt、ADNI2 Initial Visit-Cont Pt、ADNI3 Initial Visit-Cont Pt、ADNI4 Baseline - New Pt、ADNI4 Initial Visit - Cont Pt |
| m12 | ADNI1/GO Month 12、ADNI2 Year 1 Visit、ADNI3 Year 1 Visit、ADNI4 Month 12 |
| m24 | ADNI1/GO Month 24、ADNI2 Year 2 Visit、ADNI3 Year 2 Visit、Month 24 |
| m36 | ADNI1/GO Month 36、ADNI2 Year 3 Visit、ADNI3 Year 3 Visit、Month 36 |
| m48 | ADNI1/GO Month 48、ADNI2 Year 4 Visit、ADNI3 Year 4 Visit |
| m60 | ADNIGO Month 60、ADNI2 Year 5 Visit、ADNI3 Year 5 Visit |

**Description 选择说明**（同一受试者同一时间点有多个版本时）：

| Description | 选择 |
|-------------|------|
| `MPR; GradWarp; B1 Correction; N3; Scaled` | ✅ **选这个** |
| `MPR; GradWarp; B1 Correction; N3; Scaled_2` | 备用（主版本不存在时选）|
| `MPR-R; ...`（含 -R） | ❌ 跳过（这是重扫版本）|
| `MPR; GradWarp; B1 Correction; N3`（无 Scaled）| ❌ 跳过 |
| `MPR; GradWarp; B1 Correction`（无 N3） | ❌ 跳过 |
| `HarP`、`Reoriented`、`Brain Mask`、`MUSE` | ❌ 跳过（衍生产品）|

> **为什么选 N3; Scaled**：该版本已包含梯度失真校正（GradWarp）、B1 场校正、N3 偏场校正和强度缩放，与 FreeSurfer 实际运行的输入一致，下载后无需额外偏场校正。

---

## 6. 在 ADNI IDA 下载 PET 图像

步骤同 MRI，在 Advanced Image Search 中：

- **Modality = PET**
- **Description** 包含以下关键词（选择预处理版本）：
  ```
  FDG  +  Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res
  ```
- **Visit**：勾选同 MRI（bl / m12 / m24 / m36 / m48 / m60 对应的全部选项）

> 此版本已 co-registered 到对应 MRI，无需额外 PET→MRI 配准步骤。

同样 **Add to Collection → Advanced Download → NIfTI + CSV 清单**。

---

## 7. 整理图像文件

下载解压后目录结构通常为 ADNI 默认的多层嵌套，需要整理为标准命名。

### 7-A  下载清单文件位置

下载时勾选 **"Download CSV"**，保存为：
- `data/MRI_ImageCollection.csv`
- `data/PET_ImageCollection.csv`

### 7-B  整理 MRI

```python
from data.download_adni import organize_images

organize_images(
    collection_csv   = "data/MRI_ImageCollection.csv",
    raw_download_dir = "data/mri_downloaded/",   # 解压后的原始目录
    output_dir       = "data/mri_raw/",
    modality         = "MRI",
)
```

### 7-C  整理 PET

```python
from data.download_adni import organize_images

organize_images(
    collection_csv   = "data/PET_ImageCollection.csv",
    raw_download_dir = "data/pet_downloaded/",
    output_dir       = "data/pet_raw/",
    modality         = "PET",
)
```

整理后结构：
```
data/mri_raw/
  002_S_0413_bl.nii.gz
  002_S_0413_m12.nii.gz
  ...
data/pet_raw/
  002_S_0413_bl.nii.gz
  ...
```

### 7-D  DICOM 转 NIfTI（如下载的是 DICOM 格式）

```bash
# Linux
sudo apt-get install dcm2niix
# Mac
brew install dcm2niix

python -c "
from data.download_adni import convert_dicoms_to_nifti
convert_dicoms_to_nifti('data/mri_dicom/', 'data/mri_raw/')
"
```

### 7-E  检查覆盖率

```bash
python data/download_adni.py
```

预期输出：
```
=== 各访视图像覆盖数量 ===
       has_mri  has_pet
bl        1369      856
m12       1102      689
m24        934      571
...
```

---

## 8. 预处理表格数据

```bash
python data/preprocess_tabular.py
```

**输出**：`data/tabular_processed.pkl`

处理步骤：
1. 只保留 bl / m12 / m24 / m36 / m48 / m60 六个时间点
2. 排除诊断逆转受试者（MCI→CN 或 AD→MCI/CN）
3. 排除访视数 < 2 的受试者
4. 生物标志物 ÷ ICV（颅内体积归一化）
5. Z-score 标准化（在训练集上 fit，验证集 transform）
6. 性别、APOE4 one-hot 编码

**预期结果**（论文 Table II）：1369 人，5768 次访视

---

## 9. 预处理图像数据

### 9-A  下载的是 `MPR; GradWarp; B1 Correction; N3; Scaled`（推荐，无需 N4）

```bash
# 线性配准（速度快，先验证流程）
python data/preprocess_imaging.py preprocess --transform Affine

# 非线性配准（精度高，与论文 SPM12 最接近，正式实验用）
python data/preprocess_imaging.py preprocess --transform SyN
```

### 9-B  下载的是原始 MPRAGE（无 N3 标记）

```bash
# 需要加 --n4 做偏场校正
python data/preprocess_imaging.py preprocess --transform SyN --n4
```

### 9-C  PET 未 co-registered（非推荐版本）

```bash
python data/preprocess_imaging.py preprocess --transform SyN --pet_mri_ref
```

**处理步骤**：
1. [可选] N4 偏场校正（`--n4` 时执行）
2. ANTsPy 配准到 MNI152 标准脑模板（替代 SPM12 Normalise）
3. 裁剪：182×218×182 → 128×160×128
4. Min-max 归一化到 [0, 1]
5. 保存为 `.npy`

**输出**：
```
data/mri_preprocessed/{PTID}_{VISCODE}.npy
data/pet_preprocessed/{PTID}_{VISCODE}.npy
```

> 图像预处理较慢（SyN 每张约 5–15 分钟），建议多核并行或在服务器上运行。

---

## 10. 训练模型

### 完整多模态训练（MRI + PET + 表格）

```bash
# 5-fold 交叉验证（论文设置，约 100 epoch × 5 fold）
python train.py

# 仅训练某一 fold（调试时用）
python train.py --fold 0
```

### 仅用表格数据（无图像，快速验证）

```bash
python train.py --no_images
```

**超参数**（论文 Section IV.A）：

| 参数 | 值 |
|------|----|
| 优化器 | Adam |
| 学习率 | 0.002 |
| 隐藏层维度 | 128 |
| 潜在空间维度 | 256 |
| Batch size | 4 |
| 最大 Epoch | 100 |
| Early stopping patience | 20 |
| K-fold | 5 |

**输出**：`checkpoints/fold{k}/best_model.pth`

### 一键端到端流水线（数据已准备好后使用）

```bash
# 完整流水线
python run_pipeline.py --rda_dir ~/ADNIMERGE2/data

# 跳过图像预处理（仅表格）
python run_pipeline.py --rda_dir ~/ADNIMERGE2/data --skip_imaging --no_images
```

---

## 11. 评估模型

```bash
# 评估 fold 0
python evaluate.py --checkpoint checkpoints/fold0/best_model.pth --fold 0

# 评估所有 fold（脚本会自动循环）
python evaluate.py --all_folds
```

**评估指标**：

| 任务 | 指标 |
|------|------|
| 诊断预测 | Accuracy, Precision, Recall, mAUC |
| 生物标志物插补 | MAE, MRE |
| 图像重建 | MSE, PSNR |

**论文最优结果**（Table VI，全模态）：

| 指标 | 均值 ± 标准差 |
|------|-------------|
| Accuracy | 0.6222 ± 0.0128 |
| Precision | 0.6338 ± 0.0138 |
| Recall | 0.6295 ± 0.0160 |
| mAUC | 0.7899 ± 0.0130 |

---

## 常见问题

**Q：没有 GPU，能运行吗？**
A：表格模式（`--no_images`）可以在 CPU 上运行。图像模式需要 GPU。

**Q：antspyx 安装失败？**
```bash
pip install antspyx --no-build-isolation
# 或者用 conda
conda install -c aramislab antspyx
```

**Q：ADNIMERGE.csv 里某些访视的 Hippocampus 是 NaN？**
A：正常现象。该受试者该次访视没有 FreeSurfer 结果，预处理脚本会标记为缺失并在训练中插补。

**Q：图像预处理太慢？**
A：先用 `--transform Affine` 验证流程（每张约 1 分钟），正式实验再换 `SyN`。

**Q：下载的 MRI Image ID 在 IDA 搜不到？**
A：部分 IMAGEUID 对应的图像已从 IDA 下架或被替换，跳过即可（论文中有约 10% 的缺失率属于正常）。

---

## 模型架构

```
每次访视输入
├── MRI (128×160×128) ─┐
│                       ├─ M³VAE（PoE 融合）→ fused_μ (256D)
└── PET (128×160×128) ─┘
                               │
非图像特征 (13D) ──────────────┼── Imputation Module → 插补后特征
  [bio(6) + age/edu(2)         │
   + gender_oh(2) + apoe4(3)]  │
                         IRLSTM (h=128)
                          时间衰减：γ = exp(−softplus(Wδ+b))
                          改进遗忘门：g = f − sin(fπ)cos(fπ)/π
                               │
                    ┌──────────┴──────────┐
                诊断预测              生物标志物预测
             Softmax(3类)            Linear(6维)
             CN / MCI / Dementia     Ventricles ... MidTemp
```

---

## 依赖环境

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA ≥ 11.8（图像模型训练）
- ANTsPy ≥ 0.3.8（图像配准，替代 SPM12）
- pyreadr（读取 ADNIMERGE2 .rda 文件）
- nilearn（下载 MNI152 模板）

---

## 引用

```bibtex
@article{dao2025lmdpnet,
  title   = {Longitudinal Alzheimer's Disease Progression Prediction With
             Modality Uncertainty and Optimization of Information Flow},
  author  = {Dao, Duy-Phuong and Yang, Hyung-Jeong and Kim, Jahae and Ho, Ngoc-Huynh},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  volume  = {29},
  number  = {1},
  pages   = {259--272},
  year    = {2025},
  doi     = {10.1109/JBHI.2024.3472462}
}
```
