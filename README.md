# LMDP-Net 复现

**论文**：Dao et al., "Longitudinal Alzheimer's Disease Progression Prediction With Modality Uncertainty and Optimization of Information Flow", IEEE J. Biomed. Health Inform., 2025.

---

## 项目结构

```
LMDP/
├── config.py                  # 所有超参数与路径配置
├── run_pipeline.py            # 端到端一键运行脚本
├── dataset.py                 # PyTorch Dataset / DataLoader
├── train.py                   # 训练（5-fold CV）
├── evaluate.py                # 评估指标
├── requirements.txt
├── data/
│   ├── download_adni.py       # 下载说明 + 图像整理工具
│   ├── preprocess_tabular.py  # ADNIMERGE.csv 预处理
│   └── preprocess_imaging.py  # MRI/PET 图像配准+裁剪（纯Python，替代SPM12）
└── models/
    ├── m3vae.py               # M³VAE（多模态神经影像融合，含 PoE）
    ├── irlstm.py              # IRLSTM（改进遗忘门的LSTM）
    └── lmdp_net.py            # LMDP-Net 完整模型
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：`antspyx` 安装可能需要几分钟（C++ 编译）。如果失败可以用：
> ```bash
> pip install antspyx --no-build-isolation
> ```

### 2. 准备数据（见下方详细说明）

### 3. 运行

```bash
# 完整流水线（有 GPU + 图像数据）
python run_pipeline.py

# 仅用表格数据（快速验证，无需图像）
python run_pipeline.py --no_images --skip_imaging

# 仅运行训练（数据已准备好）
python train.py --fold 0
```

---

## ADNI 数据下载详细说明

### 申请权限

1. 访问 https://adni.loni.usc.edu/
2. 点击 **"Apply for Access"** → 填写研究目的（学术研究/论文复现）
3. 审核通常 1-2 个工作日

---

### 第一步：获取表格数据

#### 方案 A（推荐）：从 ADNIMERGE2 R 包 .rda 文件自动构建

如果你已下载 `ADNIMERGE2.tar.gz` 并解压到本地：

```bash
# 解压
tar -xzf ADNIMERGE2.tar.gz

# 从 .rda 文件自动生成 data/ADNIMERGE.csv
python data/build_adnimerge_from_rda.py \
    --rda_dir ~/LMDP/ADNIMERGE2/data \
    --output  data/ADNIMERGE.csv

# 先做列名诊断（可选，确认列名匹配）
python data/build_adnimerge_from_rda.py \
    --rda_dir ~/LMDP/ADNIMERGE2/data \
    --inspect_only
```

依赖：`pip install pyreadr`

合并的文件及其作用：

| 文件 | 用途 |
|------|------|
| `DXSUM.rda` | 诊断标签（CN/MCI/Dementia）|
| `PTDEMOG.rda` | 性别、教育、出生年月 |
| `APOERES.rda` | APOE4 等位基因数 |
| `REGISTRY.rda` | 各访视日期（EXAMDATE）|
| `UCSFFSX51ALL.rda` | FreeSurfer v5.1 脑区体积（ADNI1/GO）|
| `UCSFFSX.rda` | FreeSurfer 脑区体积（ADNI GO/2）|
| `UCSFFSX6.rda` | FreeSurfer v6 脑区体积（ADNI3）|
| `UCSFFSX7.rda` | FreeSurfer v7 脑区体积（所有相位）|

#### 方案 B：直接下载 ADNIMERGE.csv

登录 adni.loni.usc.edu：
```
Study Data → Key ADNI Tables and Composite Measures → ADNIMERGE → 下载 CSV
```

保存到：`data/ADNIMERGE.csv`

**论文用到的列**（均在 ADNIMERGE 中）：

| 列名 | 含义 |
|------|------|
| PTID | 受试者ID，格式 `XXX_S_XXXX` |
| VISCODE | 访视代码（`bl`, `m12`, `m24`, `m36`, `m48`, `m60`）|
| DX | 当次诊断（`CN`, `MCI`, `Dementia`）|
| AGE | 基线年龄 |
| PTGENDER | 性别 |
| PTEDUCAT | 教育年限 |
| APOE4 | APOE4 等位基因数（0/1/2）|
| Ventricles | 侧脑室体积（mm³）|
| Hippocampus | 海马体积（mm³）|
| WholeBrain | 全脑体积（mm³）|
| Entorhinal | 内嗅皮层体积（mm³）|
| Fusiform | 梭状回体积（mm³）|
| MidTemp | 中颞回体积（mm³）|
| ICV | 颅内总容积（mm³）|

---

### 第二步：下载 MRI 图像（T1-weighted MPRAGE）

#### 在 ADNI IDA 界面操作

登录 https://ida.loni.usc.edu/ → **Download → Image Collections → Advanced Image Search**

搜索条件（分4个数据集分别搜索）：

| 字段 | ADNI1 | ADNI GO | ADNI2 | ADNI3 |
|------|-------|---------|-------|-------|
| Project | ADNI1 | ADNI GO | ADNI2 | ADNI3 |
| Modality | MRI | MRI | MRI | MRI |
| Image Description | `MPRAGE` | `MPRAGE` | `Accelerated Sagittal MPRAGE` | `MPRAGE` |
| Visit | Baseline, M12, M24, M36, M48, M60 | 同左 | 同左 | 同左 |

搜索完成后：
1. **全选结果** → **Add to Collection**（命名例如 `MRI_MPRAGE_AllVisits`）
2. 进入 Collection → **Advanced Download**
3. 选择格式：**NIfTI** 或 **DICOM**（推荐 NIfTI，已转换好）
4. 下载 **图像清单 CSV**（含 Image Data ID）和图像压缩包

#### 图像目录整理

下载后运行：

```bash
python data/download_adni.py
```

或手动将图像重命名为 `{PTID}_{VISCODE}.nii.gz` 后放入：
```
data/mri_raw/
  002_S_0413_bl.nii.gz
  002_S_0413_m12.nii.gz
  ...
```

#### 通过 Image Data ID 精确下载（推荐）

ADNI IDA 下载的 CSV 清单格式如下（`MRI_ImageCollection.csv`）：

```
Subject,Group,Sex,Age,Visit,Modality,Description,Type,Acq Date,Format,Downloaded,Image Data ID
002_S_0413,MCI,M,73.4,ADNI Screening,MRI,MPRAGE,Original,2005/09/08,NiFTI,Y,I45102
002_S_0413,MCI,M,73.4,ADNI1/GO Month 12,MRI,MPRAGE,Original,2006/10/16,NiFTI,Y,I56789
...
```

**Image Data ID**（最后一列）即为每张扫描的唯一标识。用以下脚本批量匹配：

```python
from data.download_adni import organize_images

organize_images(
    collection_csv   = "data/MRI_ImageCollection.csv",   # 从 ADNI 下载的清单
    raw_download_dir = "data/mri_downloaded/",           # 下载的原始目录
    output_dir       = "data/mri_raw/",                  # 整理后的目录
    modality         = "MRI",
)
```

---

### 第三步：下载 PET 图像（FDG-PET）

#### 搜索条件

```
Modality    : PET
Description : 包含 "FDG" 且包含 "Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res"
```

> **推荐使用 ADNI 已预处理的 PET 版本**（Co-registered to MRI + Uniformly resampled）
> 可以跳过 PET-to-MRI 配准步骤，节省大量处理时间。

#### PET 图像清单 CSV 格式同 MRI，同样用 Image Data ID 匹配。

整理到：
```
data/pet_raw/
  002_S_0413_bl.nii.gz
  002_S_0413_m12.nii.gz
  ...
```

#### DICOM 转 NIfTI（如下载的是 DICOM 格式）

```bash
# 安装 dcm2niix
sudo apt-get install dcm2niix   # Linux
brew install dcm2niix           # Mac

# 批量转换（对每个受试者目录）
python -c "
from data.download_adni import convert_dicoms_to_nifti
convert_dicoms_to_nifti('data/mri_dicom/', 'data/mri_raw/')
"
```

---

### 第四步：检查数据覆盖率

```bash
python data/download_adni.py
```

输出示例：
```
=== 各访视的图像覆盖数量 ===
       has_mri  has_pet
bl        1369      856
m12       1102      689
m24        934      571
...
```

---

## 数据预处理

### 表格数据预处理

```bash
python data/preprocess_tabular.py
```

生成：`data/tabular_processed.pkl`

**处理步骤**（论文 Section IV.A）：
1. 只保留 M0, M12, M24, M36, M48, M60 六个时间点
2. 排除诊断逆转受试者（MCI→CN 或 AD→MCI/CN）
3. 排除访视数 < 2 的受试者
4. 生物标志物 ÷ ICV（颅内体积归一化）
5. Z-score 归一化（在训练集上 fit，验证集上 transform）
6. 性别、APOE4 one-hot 编码

**预期结果**（论文 Table II）：1369 人，5768 次访视

---

### 图像预处理

```bash
# 使用线性配准（速度快，推荐先试用）
python data/preprocess_imaging.py --transform Affine

# 使用非线性配准（SyN，精度高，与论文 SPM12 最接近）
python data/preprocess_imaging.py --transform SyN

# PET 未 co-registered 时（先对齐到 MRI 再到 MNI）
python data/preprocess_imaging.py --transform SyN --pet_mri_ref
```

**处理步骤**（替代论文中的 SPM12+MATLAB）：
1. ANTsPy 配准到 MNI152 标准脑模板（等价于 SPM12 Normalise）
2. 裁剪边缘：182×218×182 → 128×160×128
3. Min-max 归一化到 [0, 1]
4. 保存为 `.npy` 文件

生成：
```
data/mri_preprocessed/{PTID}_{VISCODE}.npy
data/pet_preprocessed/{PTID}_{VISCODE}.npy
```

---

## 训练

```bash
# 5-fold 交叉验证（论文设置）
python train.py

# 指定 fold 0
python train.py --fold 0

# 无图像模式（仅表格数据，快速调试）
python train.py --no_images
```

**超参数**（论文 Section IV.A）：

| 参数 | 值 |
|------|----|
| 优化器 | Adam |
| 学习率 | 0.002 |
| 隐藏层大小 | 128 |
| 潜在维度 | 256 |
| Batch size | 4 |
| 最大 Epoch | 100 |
| K-fold | 5 |

---

## 评估

```bash
python evaluate.py --checkpoint checkpoints/fold0/best_model.pth --fold 0
```

**评估指标**：

| 任务 | 指标 |
|------|------|
| 诊断预测 | Accuracy, Precision, Recall, mAUC |
| 生物标志物插补 | MAE, MRE |
| 图像重建 | MSE, PSNR |

**论文最优结果**（Table VI，使用全部模态）：
- Accuracy: 0.6222 ± 0.0128
- Precision: 0.6338 ± 0.0138
- Recall: 0.6295 ± 0.0160
- mAUC: 0.7899 ± 0.0130

---

## 模型架构

```
输入（每次访视）
├── MRI (128×160×128) ─┐
│                       ├─ M³VAE (PoE 融合) → fused_μ (256D)
└── PET (128×160×128) ─┘
                               │
非图像特征 (13D) ──────────────┼── Imputation → 插补后特征
                               │
                        IRLSTM (h=128)
                         - 时间衰减：γ = exp(-max(0, Wδ+b))
                         - 改进遗忘门：g = f - sin(fπ)cos(fπ)/π
                               │
                     ┌─────────┴─────────┐
                 诊断预测              生物标志物预测
               Softmax(3类)           Linear(6维)
```

---

## 依赖环境

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA ≥ 11.8（推荐 A6000 或 V100）
- ANTsPy ≥ 0.3.8（替代 SPM12，用于图像配准）
- 内存：≥ 32GB RAM（图像预处理），≥ 16GB GPU（训练）

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
