"""
ADNI数据下载指南与工具脚本
=================================

步骤总览
--------
1. 注册 ADNI 账户并申请数据访问权限（adni.loni.usc.edu）
2. 下载 ADNIMERGE.csv（表格数据）
3. 下载 MRI 图像（T1-weighted MPRAGE）
4. 下载 PET 图像（FDG-PET，co-registered 预处理版本）
5. 使用本脚本将下载的 CSV 清单与 ADNIMERGE 配对，生成下载批次命令

─────────────────────────────────────────────────────────────────────────────
步骤 1：注册账户
─────────────────────────────────────────────────────────────────────────────
访问：https://adni.loni.usc.edu/
点击 "Apply for Access"，填写用途（学术研究/论文复现）。
审核通常在1-2个工作日内完成。

─────────────────────────────────────────────────────────────────────────────
步骤 2：下载 ADNIMERGE.csv
─────────────────────────────────────────────────────────────────────────────
登录后：
  Study Data → Key ADNI Tables and Composite Measures → ADNIMERGE

下载后保存到：data/ADNIMERGE.csv

本论文使用的列：
  - PTID        : 受试者ID（格式如 "002_S_0413"）
  - RID         : 数字型受试者ID
  - VISCODE     : 访视代码（bl, m12, m24, m36, m48, m60）
  - EXAMDATE    : 检查日期
  - DX          : 当次诊断（CN, MCI, Dementia）
  - DX_bl       : 基线诊断
  - AGE         : 基线年龄
  - PTGENDER    : 性别（Male/Female）
  - PTEDUCAT    : 教育年限
  - APOE4       : APOE4等位基因数（0, 1, 2）
  - Ventricles  : 侧脑室体积（mm³）
  - Hippocampus : 海马体积（mm³）
  - WholeBrain  : 全脑体积（mm³）
  - Entorhinal  : 内嗅皮层体积（mm³）
  - Fusiform    : 梭状回体积（mm³）
  - MidTemp     : 中颞回体积（mm³）
  - ICV         : 颅内总容积（mm³）

─────────────────────────────────────────────────────────────────────────────
步骤 3：下载 MRI 图像（T1-weighted MPRAGE）
─────────────────────────────────────────────────────────────────────────────
登录 IDA（Image and Data Archive）：https://ida.loni.usc.edu/
  → Download → Image Collections → Advanced Image Search

搜索条件：
  - Project/Phase : ADNI1, ADNI GO, ADNI2, ADNI3（全选）
  - Modality      : MRI
  - Image Description（以下任选，覆盖全部时间段）：
      ADNI1 用: "MPRAGE" 或 "MP-RAGE"
      ADNI2/GO 用: "Accelerated Sagittal MPRAGE" 或 "ADNI_Brain_T1_MPRAGE"
      ADNI3 用: "ADNI 3D T1" 或 "MPRAGE"
  - Visit         : ADNI Screening/Baseline, Month 12, Month 24,
                    Month 36, Month 48, Month 60

→ 点击 "Search"，结果出来后 → "Add to Collection"（命名为 MRI_MPRAGE_AllVisits）
→ Collection 里点击 "1-Click Download" 或 "Advanced Download"
→ 下载 CSV 清单文件（包含所有 Image ID），用本脚本生成批量下载命令

文件格式：下载时选择 NIfTI（如果提供），否则选 DICOM 后用 dcm2niix 转换

─────────────────────────────────────────────────────────────────────────────
步骤 4：下载 PET 图像（FDG-PET，Co-Registered Preprocessed）
─────────────────────────────────────────────────────────────────────────────
建议使用 ADNI 提供的已预处理 FDG-PET 版本，节省对齐工作：

搜索条件：
  - Modality     : PET
  - Image Description 包含（优先级从高到低）：
      "FDG" + "Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res"
      或 "FDG"（只要是 FDG-PET 即可）

注意：
  - 如果 ADNI 提供了 Co-registered 版本（与对应访视的 MRI 对齐），优先使用
  - Image ID 可在下载的 CSV 清单中找到（列名 Image Data ID 或 ImageUID）

─────────────────────────────────────────────────────────────────────────────
步骤 5：图像目录结构
─────────────────────────────────────────────────────────────────────────────
将下载的 NIfTI 文件（或转换后的 NIfTI）放置如下：

  data/
  ├── mri_raw/
  │   └── {PTID}_{VISCODE}.nii.gz     # 例：002_S_0413_bl.nii.gz
  ├── pet_raw/
  │   └── {PTID}_{VISCODE}.nii.gz
  └── ADNIMERGE.csv

本脚本的 match_images_to_adnimerge() 函数可自动完成 PTID+VISCODE 的匹配和重命名。
"""

import os
import re
import pandas as pd
import shutil
from pathlib import Path


# ─── 用 Image Collection CSV 清单重命名/整理下载图像 ────────────────────────────

def parse_adni_image_collection_csv(csv_path: str) -> pd.DataFrame:
    """
    解析从 ADNI IDA 下载的 Image Collection CSV 清单。

    CSV 通常包含列：
      Subject, Group, Sex, Age, Visit, Modality, Description,
      Type, Acq Date, Format, Downloaded, Image Data ID
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # 标准化列名（ADNI 不同时期CSV格式略有差异）
    col_map = {
        "Subject": "PTID",
        "subject": "PTID",
        "Visit": "VISIT",
        "visit": "VISIT",
        "Modality": "MODALITY",
        "Image Data ID": "IMAGE_ID",
        "image_data_id": "IMAGE_ID",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return df


def map_visit_to_viscode(visit_str: str) -> str:
    """
    将 IDA 中的 Visit 字符串映射到 ADNIMERGE 的 VISCODE 格式。

    ADNI IDA Visit名例子：
      "ADNI Screening" / "ADNI1/GO Month 0" → bl
      "ADNI1/GO Month 12" → m12
      "ADNI2 Year 1 Visit" / "Month 12" → m12
    """
    v = visit_str.lower().strip()
    if any(x in v for x in ["screen", "baseline", "month 0", "year 0", "initial"]):
        return "bl"
    m = re.search(r"month\s*(\d+)", v)
    if m:
        return f"m{m.group(1)}"
    y = re.search(r"year\s*(\d+)", v)
    if y:
        months = int(y.group(1)) * 12
        return f"m{months}"
    return ""


def build_rename_map(collection_csv: str, dicom_download_dir: str) -> dict:
    """
    返回 {原始路径: 目标路径} 的重命名字典，
    用于将 ADNI 下载后的散乱目录整理为 {PTID}_{VISCODE}.nii.gz 格式。

    ADNI 下载后的目录结构通常是：
      download_dir/{SubjectID}/{Visit}/{ImageID}/xxx.nii / xxx.dcm
    """
    df = parse_adni_image_collection_csv(collection_csv)
    rename_map = {}

    for _, row in df.iterrows():
        ptid = str(row.get("PTID", "")).strip().replace(" ", "_")
        visit_str = str(row.get("VISIT", "")).strip()
        img_id = str(row.get("IMAGE_ID", "")).strip()
        viscode = map_visit_to_viscode(visit_str)
        if not viscode or not ptid:
            continue

        # 在下载目录中查找对应文件
        pattern_dirs = [
            os.path.join(dicom_download_dir, ptid),
            os.path.join(dicom_download_dir, ptid.replace("_", " ")),
        ]
        for base in pattern_dirs:
            if not os.path.exists(base):
                continue
            for root, dirs, files in os.walk(base):
                for f in files:
                    if f.endswith((".nii", ".nii.gz")) and img_id in root:
                        src = os.path.join(root, f)
                        dst_name = f"{ptid}_{viscode}.nii.gz"
                        rename_map[src] = dst_name
                        break
    return rename_map


def organize_images(collection_csv: str,
                    raw_download_dir: str,
                    output_dir: str,
                    modality: str = "MRI") -> None:
    """
    将 ADNI 下载的图像按 {PTID}_{VISCODE}.nii.gz 命名，复制到 output_dir。

    参数
    ----
    collection_csv   : ADNI IDA 下载的 CSV 清单
    raw_download_dir : ADNI 下载的图像根目录
    output_dir       : 整理后图像的输出目录（如 data/mri_raw/ 或 data/pet_raw/）
    modality         : "MRI" 或 "PET"
    """
    os.makedirs(output_dir, exist_ok=True)
    rename_map = build_rename_map(collection_csv, raw_download_dir)

    copied, skipped = 0, 0
    for src, dst_name in rename_map.items():
        dst = os.path.join(output_dir, dst_name)
        if os.path.exists(dst):
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
        print(f"  [{modality}] {os.path.basename(src)} → {dst_name}")

    print(f"\n{modality}: 复制 {copied} 个文件，跳过 {skipped} 个已存在文件。")


# ─── DICOM → NIfTI 转换（需要安装 dcm2niix）────────────────────────────────────

def convert_dicoms_to_nifti(dicom_root: str, output_dir: str) -> None:
    """
    遍历 dicom_root 中的每个受试者目录，使用 dcm2niix 转换为 NIfTI。

    安装 dcm2niix:
      Linux : sudo apt-get install dcm2niix
      MacOS : brew install dcm2niix
      pip   : pip install dcm2niix
    """
    import subprocess
    os.makedirs(output_dir, exist_ok=True)
    for subject_dir in sorted(Path(dicom_root).iterdir()):
        if not subject_dir.is_dir():
            continue
        cmd = [
            "dcm2niix",
            "-z", "y",          # gzip 压缩
            "-f", "%n_%v",      # 文件命名：subject_visit
            "-o", output_dir,
            str(subject_dir),
        ]
        print(f"Converting {subject_dir.name} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [警告] dcm2niix 出错: {result.stderr[:200]}")


# ─── 从 ADNIMERGE.csv 获取具有图像数据的受试者列表 ─────────────────────────────

def get_subjects_with_images(adnimerge_path: str,
                             mri_dir: str,
                             pet_dir: str) -> pd.DataFrame:
    """
    返回 ADNIMERGE 中那些在 mri_dir 或 pet_dir 中有实际图像文件的行。
    可以用这个来了解数据覆盖率。
    """
    df = pd.read_csv(adnimerge_path, low_memory=False)
    visit_codes = ["bl", "m12", "m24", "m36", "m48", "m60"]
    df = df[df["VISCODE"].isin(visit_codes)].copy()

    mri_files = {f.stem.replace(".nii", ""): f for f in Path(mri_dir).glob("*.nii*")} \
        if os.path.exists(mri_dir) else {}
    pet_files = {f.stem.replace(".nii", ""): f for f in Path(pet_dir).glob("*.nii*")} \
        if os.path.exists(pet_dir) else {}

    def has_file(ptid, viscode, file_dict):
        key = f"{ptid}_{viscode}"
        return key in file_dict

    df["has_mri"] = df.apply(lambda r: has_file(r["PTID"], r["VISCODE"], mri_files), axis=1)
    df["has_pet"] = df.apply(lambda r: has_file(r["PTID"], r["VISCODE"], pet_files), axis=1)

    coverage = df.groupby("VISCODE")[["has_mri", "has_pet"]].sum()
    print("\n=== 各访视的图像覆盖数量 ===")
    print(coverage.to_string())
    return df


# ─── 快速检查（脚本直接运行时） ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ADNI 数据下载辅助工具")
    print("=" * 60)
    print(__doc__)

    # 示例：检查已有图像覆盖率
    adnimerge = "data/ADNIMERGE.csv"
    if os.path.exists(adnimerge):
        info_df = get_subjects_with_images(adnimerge, "data/mri_raw", "data/pet_raw")
        print(f"\n总记录数（目标访视）: {len(info_df)}")
        print(f"有MRI的记录: {info_df['has_mri'].sum()}")
        print(f"有PET的记录: {info_df['has_pet'].sum()}")
    else:
        print(f"[!] 未找到 {adnimerge}，请先下载 ADNIMERGE.csv")
