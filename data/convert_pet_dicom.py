"""
Convert ADNI PET DICOM files to NIfTI and organize into pet_raw/.

用法
----
# 第一步：解压 zip 并转换（自动处理两个 zip）
python data/convert_pet_dicom.py \
    --zip_dir  data/pet_downloaded/ \
    --csv      data/pet_download_list.csv \
    --out_dir  data/pet_raw/

# 如果已经解压到 pet_downloaded/ADNI/，跳过解压直接转换：
python data/convert_pet_dicom.py \
    --adni_dir data/pet_downloaded/ADNI/ \
    --csv      data/pet_download_list.csv \
    --out_dir  data/pet_raw/
"""

import os
import re
import subprocess
import tempfile
import zipfile
import shutil
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def build_uid_map(csv_path: str) -> dict:
    dl = pd.read_csv(csv_path)
    dl["IMAGEUID"] = pd.to_numeric(dl["IMAGEUID"], errors="coerce").astype("Int64")
    dl = dl.dropna(subset=["IMAGEUID", "PTID", "VISCODE"])
    return {int(row["IMAGEUID"]): (str(row["PTID"]), str(row["VISCODE"]))
            for _, row in dl.iterrows()}


def extract_image_id(path: str):
    """从路径中提取 Image ID（如 .../I1592036/... → 1592036）。"""
    # 优先匹配末尾的 ImageID 目录
    m = re.search(r"[/\\]I(\d+)(?:[/\\]|$)", str(path))
    if m is None:
        m = re.search(r"I(\d+)", os.path.basename(str(path)))
    return int(m.group(1)) if m else None


def find_dicom_dirs(adni_root: str) -> list:
    """找到所有包含 .dcm 文件的 ImageID 目录。"""
    dicom_dirs = []
    for dirpath, _, files in os.walk(adni_root):
        if any(f.endswith(".dcm") for f in files):
            dicom_dirs.append(dirpath)
    return dicom_dirs


def convert_one_dir(dicom_dir: str, dst_path: str) -> bool:
    """
    用 dcm2niix 转换一个 DICOM 目录，输出到 dst_path（.nii.gz）。
    返回 True 表示成功。
    """
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ["dcm2niix", "-z", "y", "-f", "pet_output", "-o", tmp, dicom_dir],
            capture_output=True, text=True
        )
        nii_files = sorted(Path(tmp).glob("*.nii.gz")) or sorted(Path(tmp).glob("*.nii"))
        if not nii_files:
            return False
        # 取第一个（通常只有一个）
        shutil.move(str(nii_files[0]), dst_path)
        return True


def extract_zips(zip_dir: str, out_dir: str) -> None:
    """解压 zip_dir 里所有 zip 到 out_dir（支持断点续传，不重复解压已有文件）。"""
    zip_files = list(Path(zip_dir).glob("*.zip"))
    print(f"找到 {len(zip_files)} 个 zip 文件")
    for zp in zip_files:
        print(f"\n解压: {zp.name}")
        with zipfile.ZipFile(str(zp), "r") as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            for member in tqdm(members, desc=zp.name):
                dst = os.path.join(out_dir, member.filename)
                if os.path.exists(dst):
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                zf.extract(member, out_dir)


def process(adni_dir: str, csv_path: str, out_dir: str) -> None:
    uid_map = build_uid_map(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    dicom_dirs = find_dicom_dirs(adni_dir)
    print(f"找到 {len(dicom_dirs)} 个 DICOM 目录")

    converted = skipped_exists = not_in_list = failed = 0
    for d in tqdm(dicom_dirs, desc="DICOM→NIfTI"):
        img_id = extract_image_id(d)
        if img_id is None or img_id not in uid_map:
            not_in_list += 1
            continue

        ptid, viscode = uid_map[img_id]
        dst = os.path.join(out_dir, f"{ptid}_{viscode}.nii.gz")

        if os.path.exists(dst):
            skipped_exists += 1
            continue

        ok = convert_one_dir(d, dst)
        if ok:
            converted += 1
        else:
            failed += 1
            print(f"  [警告] 转换失败: {d}")

    print(f"\n转换: {converted}  |  已存在(跳过): {skipped_exists}  "
          f"|  不在下载清单: {not_in_list}  |  失败: {failed}")
    print(f"输出目录: {out_dir}")
    print(f"文件数:   {len(list(Path(out_dir).glob('*.nii*')))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADNI PET DICOM → NIfTI 转换整理工具")
    parser.add_argument("--zip_dir",  help="包含 PET zip 文件的目录（自动解压）")
    parser.add_argument("--adni_dir", help="已解压的 ADNI/ 目录（跳过解压）")
    parser.add_argument("--csv",      default="data/pet_download_list.csv",
                        help="pet_download_list.csv 路径")
    parser.add_argument("--out_dir",  required=True, help="输出目录（pet_raw/）")
    parser.add_argument("--keep_dicom", action="store_true",
                        help="保留解压的 DICOM 文件（默认转换后删除）")
    args = parser.parse_args()

    if args.adni_dir:
        adni_dir = args.adni_dir
    elif args.zip_dir:
        adni_dir = os.path.join(args.zip_dir, "ADNI")
        extract_zips(args.zip_dir, args.zip_dir)
    else:
        parser.error("需要 --zip_dir 或 --adni_dir 之一")

    process(adni_dir, args.csv, args.out_dir)

    if args.zip_dir and not args.keep_dicom and os.path.exists(adni_dir):
        print(f"\n删除 DICOM 目录: {adni_dir}")
        shutil.rmtree(adni_dir)
        print("完成。")
