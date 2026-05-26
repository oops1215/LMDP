"""
MRI / PET 图像预处理（全 Python 实现，替代论文中的 SPM12+MATLAB）

流程（对应论文 Section IV.A）：
  1. 加载 NIfTI 图像
  2. [可选] N4 偏场校正（用于原始/轻度处理 MRI，已有 N3 的可跳过）
  3. 使用 ANTsPy 配准到 MNI152 标准脑模板（替代 SPM12 normalization）
  4. 裁剪到 128×160×128（去除边缘无关区域）
  5. Min-max 归一化到 [0, 1]
  6. 保存为 .npy 文件

─────────────────────────────────────────────────────────────
  ADNI MRI 应下载哪种预处理版本？
─────────────────────────────────────────────────────────────

  ADNI IDA 中同一次扫描通常有多个版本，按处理程度从低到高：

  ① 原始    "MPRAGE"  / "Accelerated Sagittal MPRAGE"
      - 只有最原始的 DICOM 数据，没有任何后处理
      - 若下载此类型，必须开启 --n4（偏场校正）

  ② 推荐 ★  "MPR; GradWarp; B1 Correction; N3; Scaled"
      - GradWarp：梯度非线性失真校正（Scanner-level）
      - B1 Correction：B1 场不均匀校正
      - N3（非均匀强度校正）：已做偏场校正，相当于 N4
      - Scaled：强度重新缩放
      - 论文中 FreeSurfer (UCSFFSX*) 就在这个版本上运行的
      - 下载此版本后 **无需** 开启 --n4，直接配准即可

  核心原则：**IMAGEUID（Image Data ID）与 UCSFFSX*.rda 中记录的一致**
  使用 data/generate_mri_download_list.py 提取 IMAGEUID，
  在 IDA 按 Image Data ID 精确匹配，确保 MRI 与 FreeSurfer 体积一一对应。

─────────────────────────────────────────────────────────────
  两种下载版本的预处理差异
─────────────────────────────────────────────────────────────

  | 步骤              | 原始 MPRAGE     | N3-Scaled（推荐）|
  |-------------------|-----------------|-----------------|
  | N4 偏场校正       | 必须（--n4）    | 跳过            |
  | ANTsPy 配准       | 相同            | 相同            |
  | 裁剪 128×160×128  | 相同            | 相同            |
  | Min-max 归一化    | 相同            | 相同            |

  无论哪种，ANTsPy 配准流程完全相同；唯一区别是是否需要额外的偏场校正。

─────────────────────────────────────────────────────────────
  PET 图像推荐下载版本
─────────────────────────────────────────────────────────────

  Description 选择：
    "FDG" + "Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res"
  这是 ADNI 提供的标准化 FDG-PET（已 co-reg 到对应 MRI），可跳过 PET→MRI 配准。

─────────────────────────────────────────────────────────────
  依赖安装
─────────────────────────────────────────────────────────────
  pip install antspyx nibabel nilearn tqdm

MNI152 模板下载（选其一）：
  方法 A（推荐）：使用 nilearn 自动下载
    from nilearn.datasets import fetch_icbm152_2009
    mni = fetch_icbm152_2009()  # 自动保存到 ~/nilearn_data/

  方法 B：手动下载 FSL 的 MNI152_T1_1mm.nii.gz
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases
    或 ANTs 自带模板

  本脚本默认使用方法 A（nilearn）。
"""

import os
import sys
import glob
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# ─── N4 偏场校正 ─────────────────────────────────────────────────────────────

def apply_n4_bias_correction(image_path: str) -> "ants.ANTsImage":
    """
    使用 ANTsPy 对 MRI 做 N4 偏场校正（ants.n4_bias_field_correction）。

    何时使用：
      - 下载的是原始 MPRAGE 或 "Accelerated Sagittal MPRAGE"（无 N3 标记）
    何时跳过：
      - 下载的是 "MPR; GradWarp; B1 Correction; N3; Scaled"（已含 N3 校正）

    N4 是 N3 的改进版本，ADNI 的 N3 与 ANTsPy N4 效果接近；
    对 N3-Scaled 图像再做 N4 不会造成损害但也没有必要。
    """
    import ants
    img = ants.image_read(image_path).clone("float")
    img_corrected = ants.n4_bias_field_correction(img)
    return img_corrected


# ─── 获取 MNI152 模板路径 ────────────────────────────────────────────────────

def get_mni152_template() -> str:
    """
    优先使用 Config.MNI152_TEMPLATE，其次通过 nilearn 自动下载。
    返回 NIfTI 文件路径（字符串）。
    """
    if os.path.exists(Config.MNI152_TEMPLATE):
        return Config.MNI152_TEMPLATE

    # 尝试 nilearn
    try:
        from nilearn.datasets import fetch_icbm152_2009
        dataset = fetch_icbm152_2009()
        template_path = dataset["t1"]
        print(f"使用 nilearn MNI152 模板: {template_path}")
        return template_path
    except Exception as e:
        print(f"[警告] nilearn 获取模板失败: {e}")

    # 尝试 ANTs 内置模板
    try:
        import ants
        template_path = ants.get_ants_data("mni")
        if template_path and os.path.exists(template_path):
            print(f"使用 ANTs 内置 MNI152 模板: {template_path}")
            return template_path
    except Exception as e:
        print(f"[警告] ANTs 模板获取失败: {e}")

    raise FileNotFoundError(
        "找不到 MNI152 模板。请安装 nilearn (`pip install nilearn`) "
        "或手动下载 MNI152_T1_1mm.nii.gz 到 " + Config.MNI152_TEMPLATE
    )


# ─── ANTs 配准到 MNI152 ──────────────────────────────────────────────────────

def register_to_mni152(image_path: str,
                        template_path: str,
                        type_of_transform: str = "SyN",
                        moving_image=None) -> np.ndarray:
    """
    将输入图像配准到 MNI152 空间。
    使用 SyN（非线性）配准，等同于 SPM12 的 Normalise(Write) 步骤。

    参数
    ----
    image_path        : 输入 NIfTI 文件路径（当 moving_image 为 None 时使用）
    template_path     : MNI152 模板 NIfTI 路径
    type_of_transform : "SyN"（推荐）或 "Affine"（速度更快）
    moving_image      : 已加载的 ANTsImage（例如 N4 校正后的结果），
                        若提供则忽略 image_path

    返回
    ----
    numpy array, shape = (182, 218, 182)，已配准的图像数据
    """
    import ants

    template = ants.image_read(template_path)
    if moving_image is not None:
        moving = moving_image
    else:
        moving = ants.image_read(image_path).clone("float")

    reg = ants.registration(
        fixed=template,
        moving=moving,
        type_of_transform=type_of_transform,
        verbose=False,
    )
    registered = reg["warpedmovout"]
    return registered.numpy()


def register_pet_via_mri(pet_path: str,
                          mri_path: str,
                          template_path: str) -> np.ndarray:
    """
    先将 PET 配准到同受试者的 MRI（步骤1），
    再将 MRI 配准到 MNI152（步骤2），
    最后将步骤1的变换叠加到步骤2上，得到 PET → MNI152 的配准结果。

    如果 ADNI 已提供 co-registered PET（推荐使用此类型），
    则可以跳过 PET→MRI 步骤，直接走 MRI→MNI152。
    """
    import ants

    template = ants.image_read(template_path)
    mri      = ants.image_read(mri_path).clone("float")
    pet      = ants.image_read(pet_path).clone("float")

    # 步骤1：PET → MRI（刚体配准）
    pet2mri_reg = ants.registration(
        fixed=mri,
        moving=pet,
        type_of_transform="Rigid",
        verbose=False,
    )

    # 步骤2：MRI → MNI152（SyN）
    mri2mni_reg = ants.registration(
        fixed=template,
        moving=mri,
        type_of_transform="SyN",
        verbose=False,
    )

    # 组合变换：PET → MRI → MNI152
    pet_in_mni = ants.apply_transforms(
        fixed=template,
        moving=pet,
        transformlist=mri2mni_reg["fwdtransforms"] + pet2mri_reg["fwdtransforms"],
    )
    return pet_in_mni.numpy()


# ─── 裁剪与归一化 ────────────────────────────────────────────────────────────

def crop_image(img_arr: np.ndarray,
               start: tuple = Config.CROP_START,
               end: tuple   = Config.CROP_END) -> np.ndarray:
    """
    从配准后的 182×218×182 图像中裁剪出 128×160×128 区域（去除边缘无关脑外区域）。
    """
    return img_arr[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


def minmax_normalize(img_arr: np.ndarray) -> np.ndarray:
    """Min-max 归一化到 [0, 1]。"""
    vmin = img_arr.min()
    vmax = img_arr.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(img_arr, dtype=np.float32)
    return ((img_arr - vmin) / (vmax - vmin)).astype(np.float32)


# ─── 处理单张图像（完整流水线）──────────────────────────────────────────────

def preprocess_single_image(image_path: str,
                             template_path: str,
                             output_path: str,
                             transform_type: str = "SyN",
                             apply_n4: bool = False) -> bool:
    """
    对单张 MRI 或 PET 图像执行完整预处理，保存为 .npy 文件。
    返回 True 表示成功，False 表示跳过（已存在）或出错。

    参数
    ----
    apply_n4  : 是否先做 N4 偏场校正。
                True  → 适合原始 MPRAGE（无 N3 校正）
                False → 适合 "MPR; GradWarp; B1 Correction; N3; Scaled"（已含 N3）
    """
    if os.path.exists(output_path):
        return True  # 已存在，跳过

    try:
        # 1. [可选] N4 偏场校正
        moving_image = None
        if apply_n4:
            moving_image = apply_n4_bias_correction(image_path)

        # 2. 配准到 MNI152
        registered = register_to_mni152(image_path, template_path, transform_type,
                                         moving_image=moving_image)
        # 3. 裁剪
        cropped = crop_image(registered)
        assert cropped.shape == Config.TARGET_SHAPE, \
            f"裁剪后形状异常: {cropped.shape} != {Config.TARGET_SHAPE}"
        # 4. 归一化
        normalized = minmax_normalize(cropped)
        # 5. 保存
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.save(output_path, normalized)
        return True
    except Exception as e:
        print(f"  [错误] 处理 {os.path.basename(image_path)} 时出错: {e}")
        return False


def preprocess_pet_with_mri_reference(pet_path: str,
                                       mri_path: str,
                                       template_path: str,
                                       output_path: str) -> bool:
    """对 PET 图像使用 MRI 参考进行预处理（适用于未 co-registered 的 PET）。"""
    if os.path.exists(output_path):
        return True
    try:
        registered = register_pet_via_mri(pet_path, mri_path, template_path)
        cropped    = crop_image(registered)
        normalized = minmax_normalize(cropped)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.save(output_path, normalized)
        return True
    except Exception as e:
        print(f"  [错误] PET 处理失败 {os.path.basename(pet_path)}: {e}")
        return False


# ─── 批量处理所有受试者 ─────────────────────────────────────────────────────

def preprocess_all(mri_raw_dir: str   = Config.MRI_RAW_DIR,
                   pet_raw_dir: str   = Config.PET_RAW_DIR,
                   mri_out_dir: str   = Config.MRI_PREP_DIR,
                   pet_out_dir: str   = Config.PET_PREP_DIR,
                   transform_type: str = "SyN",
                   pet_use_mri_ref: bool = False,
                   apply_n4: bool = False) -> None:
    """
    批量预处理所有 MRI 和 PET 图像。

    参数
    ----
    mri_raw_dir      : 原始 MRI NIfTI 文件目录（{PTID}_{VISCODE}.nii.gz）
    pet_raw_dir      : 原始 PET NIfTI 文件目录
    mri_out_dir      : 预处理后 MRI .npy 文件输出目录
    pet_out_dir      : 预处理后 PET .npy 文件输出目录
    transform_type   : "SyN"（精度高，慢）或 "Affine"（速度快，精度略低）
    pet_use_mri_ref  : True = PET 先对齐到 MRI 再到 MNI（适合未 co-reg 的 PET）
    apply_n4         : True = 对 MRI 做 N4 偏场校正（适合原始 MPRAGE，无 N3 校正）
                       False（默认）= 跳过，适合 "MPR; GradWarp; B1 Correction; N3; Scaled"
    """
    os.makedirs(mri_out_dir, exist_ok=True)
    os.makedirs(pet_out_dir, exist_ok=True)

    template_path = get_mni152_template()
    print(f"MNI152 模板: {template_path}")
    if apply_n4:
        print("N4 偏场校正: 已启用（适合原始 MPRAGE）")
    else:
        print("N4 偏场校正: 已跳过（适合 N3-Scaled 版本，推荐下载此版本）")

    # ── 处理 MRI ─────────────────────────────────────────────────────────────
    mri_files = sorted(glob.glob(os.path.join(mri_raw_dir, "*.nii*")))
    print(f"\n找到 {len(mri_files)} 个 MRI 文件，开始预处理 ...")

    mri_ok, mri_fail = 0, 0
    for mri_path in tqdm(mri_files, desc="MRI 预处理"):
        fname  = os.path.splitext(os.path.basename(mri_path))[0].replace(".nii", "")
        outpath = os.path.join(mri_out_dir, fname + ".npy")
        ok = preprocess_single_image(mri_path, template_path, outpath,
                                      transform_type, apply_n4=apply_n4)
        if ok:
            mri_ok += 1
        else:
            mri_fail += 1

    print(f"MRI 完成: {mri_ok} 成功，{mri_fail} 失败")

    # ── 处理 PET ─────────────────────────────────────────────────────────────
    pet_files = sorted(glob.glob(os.path.join(pet_raw_dir, "*.nii*")))
    print(f"\n找到 {len(pet_files)} 个 PET 文件，开始预处理 ...")

    pet_ok, pet_fail = 0, 0
    for pet_path in tqdm(pet_files, desc="PET 预处理"):
        fname   = os.path.splitext(os.path.basename(pet_path))[0].replace(".nii", "")
        outpath = os.path.join(pet_out_dir, fname + ".npy")

        if pet_use_mri_ref:
            # 找对应的 MRI 文件
            mri_ref = os.path.join(mri_raw_dir, os.path.basename(pet_path))
            if not os.path.exists(mri_ref):
                # 尝试 .nii 后缀
                mri_ref = mri_ref.replace(".nii.gz", ".nii")
            if os.path.exists(mri_ref):
                ok = preprocess_pet_with_mri_reference(
                    pet_path, mri_ref, template_path, outpath)
            else:
                # 没有对应 MRI，直接配准到 MNI152
                ok = preprocess_single_image(
                    pet_path, template_path, outpath, transform_type)
        else:
            # ADNI Co-registered PET：直接配准到 MNI152（PET 无需 N4）
            ok = preprocess_single_image(
                pet_path, template_path, outpath, transform_type)

        if ok:
            pet_ok += 1
        else:
            pet_fail += 1

    print(f"PET 完成: {pet_ok} 成功，{pet_fail} 失败")


# ─── IDA 下载整理：按 Image ID 选出正确文件，重命名为 PTID_VISCODE.nii.gz ────

def organize_mri_from_ida(
        download_dir: str,
        output_dir: str,
        download_list_csv: str = "data/mri_download_list.csv",
        dry_run: bool = False,
) -> None:
    """
    从 ADNI IDA 下载目录中，只挑选 mri_download_list.csv 里记录的 Image ID，
    重命名/复制到 output_dir/{PTID}_{VISCODE}.nii.gz。

    IDA 批量下载会包含 Repeat 扫描和多余访视，本函数通过 Image ID 精确过滤，
    只保留 FreeSurfer 实际使用的那张图。

    ADNI NIfTI 文件名或目录中包含 "I{IMAGEUID}" 模式，例如：
      - 002_S_0413_MR_2005-09-08_I45102.nii
      - .../I45102/file.nii

    参数
    ----
    download_dir      : IDA 解压后的根目录（会递归搜索 .nii / .nii.gz）
    output_dir        : 整理后的输出目录（preprocess_imaging 的 mri_raw_dir）
    download_list_csv : generate_mri_download_list.py 生成的 CSV，含 PTID/VISCODE/IMAGEUID
    dry_run           : True = 只打印操作，不复制文件（先检查再实际运行）
    """
    import re
    import shutil
    import pandas as pd

    dl = pd.read_csv(download_list_csv)
    # 建立 IMAGEUID → (PTID, VISCODE) 的查找表
    dl["IMAGEUID"] = pd.to_numeric(dl["IMAGEUID"], errors="coerce").astype("Int64")
    dl = dl.dropna(subset=["IMAGEUID", "PTID", "VISCODE"])
    uid_map = {int(row["IMAGEUID"]): (str(row["PTID"]), str(row["VISCODE"]))
               for _, row in dl.iterrows()}
    print(f"下载清单: {len(uid_map)} 条 Image ID")

    # 递归找所有 NIfTI
    all_nii = []
    for root, _, files in os.walk(download_dir):
        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                all_nii.append(os.path.join(root, f))
    print(f"下载目录中找到 {len(all_nii)} 个 NIfTI 文件")

    os.makedirs(output_dir, exist_ok=True)
    copied, skipped_repeat, skipped_exists = 0, 0, 0

    for filepath in sorted(all_nii):
        # 从文件名或路径中提取 Image ID（格式：I\d+）
        match = re.search(r"[_/\\]I(\d+)", filepath)
        if match is None:
            # 有些文件名格式不同，尝试仅数字模式
            match = re.search(r"I(\d+)", os.path.basename(filepath))
        if match is None:
            print(f"  [跳过] 无法提取 Image ID: {os.path.basename(filepath)}")
            continue

        image_id = int(match.group(1))
        if image_id not in uid_map:
            skipped_repeat += 1  # Repeat 或不在目标访视列表里
            continue

        ptid, viscode = uid_map[image_id]
        ext = ".nii.gz" if filepath.endswith(".nii.gz") else ".nii"
        out_name = f"{ptid}_{viscode}{ext}"
        out_path  = os.path.join(output_dir, out_name)

        if os.path.exists(out_path):
            skipped_exists += 1
            continue

        if dry_run:
            print(f"  [DRY] {os.path.basename(filepath)} → {out_name}")
        else:
            shutil.copy2(filepath, out_path)
        copied += 1

    action = "将复制" if dry_run else "已复制"
    print(f"\n{action}: {copied}  |  跳过(重复/无关): {skipped_repeat}  |  已存在: {skipped_exists}")
    if dry_run:
        print("（dry_run 模式，未实际操作。去掉 --dry_run 后正式运行）")


# ─── 验证单张图像（可视化检查）──────────────────────────────────────────────

def visualize_preprocessed(npy_path: str, slice_idx: int = 64) -> None:
    """可视化预处理后的图像（轴位切片）。"""
    import matplotlib.pyplot as plt
    img = np.load(npy_path)
    print(f"图像形状: {img.shape}, 值范围: [{img.min():.3f}, {img.max():.3f}]")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img[slice_idx, :, :], cmap="gray"); axes[0].set_title(f"Axial (z={slice_idx})")
    axes[1].imshow(img[:, slice_idx, :], cmap="gray"); axes[1].set_title(f"Coronal (y={slice_idx})")
    axes[2].imshow(img[:, :, slice_idx], cmap="gray"); axes[2].set_title(f"Sagittal (x={slice_idx})")
    plt.suptitle(os.path.basename(npy_path))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LMDP-Net 图像预处理")
    subparsers = parser.add_subparsers(dest="command")

    # ── 子命令 1: organize（整理 IDA 下载）──────────────────────────────────
    p_org = subparsers.add_parser("organize",
        help="从 IDA 下载目录按 Image ID 挑出正确文件，重命名到 mri_raw/")
    p_org.add_argument("--download_dir", required=True,
                       help="IDA 解压后的根目录")
    p_org.add_argument("--output_dir", default=Config.MRI_RAW_DIR,
                       help="整理后的输出目录（默认 data/mri_raw）")
    p_org.add_argument("--download_list", default="data/mri_download_list.csv",
                       help="generate_mri_download_list.py 生成的 CSV")
    p_org.add_argument("--dry_run", action="store_true",
                       help="只打印操作，不复制文件")

    # ── 子命令 2: preprocess（预处理）───────────────────────────────────────
    p_pre = subparsers.add_parser("preprocess",
        help="对 mri_raw/ 中的 NIfTI 文件做配准/裁剪/归一化")
    p_pre.add_argument("--mri_dir",  default=Config.MRI_RAW_DIR)
    p_pre.add_argument("--pet_dir",  default=Config.PET_RAW_DIR)
    p_pre.add_argument("--transform", default="SyN",
                       choices=["SyN", "Affine"],
                       help="SyN=精度高/慢，Affine=速度快/精度略低")
    p_pre.add_argument("--pet_mri_ref", action="store_true",
                       help="PET 先对齐到 MRI 再到 MNI（未 co-reg PET 用此选项）")
    p_pre.add_argument("--n4", action="store_true",
                       help="对 MRI 做 N4 偏场校正（原始 MPRAGE 使用，N3-Scaled 无需）")

    args = parser.parse_args()

    if args.command == "organize":
        organize_mri_from_ida(
            download_dir=args.download_dir,
            output_dir=args.output_dir,
            download_list_csv=args.download_list,
            dry_run=args.dry_run,
        )
    elif args.command == "preprocess":
        preprocess_all(
            mri_raw_dir=args.mri_dir,
            pet_raw_dir=args.pet_dir,
            transform_type=args.transform,
            pet_use_mri_ref=args.pet_mri_ref,
            apply_n4=args.n4,
        )
    else:
        parser.print_help()
