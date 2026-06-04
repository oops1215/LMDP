"""
从 ADNI zip 文件流式提取 MRI NIfTI，按 mri_download_list.csv 重命名，
一次只解压一个文件，最小化临时磁盘占用。

两种用法
--------
1. 整理已解压的 ADNI/ 目录（存量文件）：
   python data/stream_extract_mri.py organize \\
       --src_dir  data/mri_downloaded/ADNI/ \\
       --csv      data/mri_download_list.csv \\
       --out_dir  /mnt/d/LMDP_data/mri_raw/ \\
       --move          # 移动而非复制（节省空间，ADNI/ 文件会被删除）

2. 流式从 zip 提取（不需要 zip × 4 的磁盘空间）：
   python data/stream_extract_mri.py from_zip \\
       --zip     'data/mri_downloaded/LMDP_MRILAST (1).zip' \\
       --csv      data/mri_download_list.csv \\
       --out_dir  /mnt/d/LMDP_data/mri_raw/

完整工作流（存储紧张时）：
  # 第一步：整理已解压文件并移到 D:
  python data/stream_extract_mri.py organize --src_dir data/mri_downloaded/ADNI/ \\
      --out_dir /mnt/d/LMDP_data/mri_raw/ --move

  # 第二步：流式解压第二个 zip 到 D:（每次只占 ~50MB WSL 临时空间）
  python data/stream_extract_mri.py from_zip \\
      --zip 'data/mri_downloaded/LMDP_MRILAST (1).zip' \\
      --out_dir /mnt/d/LMDP_data/mri_raw/

  # 第三步：预处理（见 preprocess_imaging.py preprocess 子命令）
  python data/preprocess_imaging.py preprocess \\
      --mri_dir /mnt/d/LMDP_data/mri_raw/ \\
      --mri_out_dir /mnt/d/LMDP_data/mri_preprocessed/ \\
      --transform Affine --delete_source
"""

import os
import re
import sys
import zipfile
import shutil
import tempfile
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def build_uid_map(csv_path: str) -> dict:
    """读取 mri_download_list.csv，构建 {image_id: (ptid, viscode)} 字典。"""
    dl = pd.read_csv(csv_path)
    dl["IMAGEUID"] = pd.to_numeric(dl["IMAGEUID"], errors="coerce").astype("Int64")
    dl = dl.dropna(subset=["IMAGEUID", "PTID", "VISCODE"])
    return {int(row["IMAGEUID"]): (str(row["PTID"]), str(row["VISCODE"]))
            for _, row in dl.iterrows()}


def extract_image_id(path: str):
    """从 ADNI 文件名/路径中提取数字 Image ID（如 I143685 → 143685）。"""
    m = re.search(r"[/_\\]I(\d+)", path)
    if m is None:
        m = re.search(r"I(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def organize_from_dir(src_dir: str, csv_path: str, out_dir: str,
                       move: bool = False) -> None:
    """
    整理已解压的 ADNI/ 目录 → out_dir/{PTID}_{VISCODE}.nii。

    参数
    ----
    move : True = 移动文件（节省空间，原文件会被删除）
           False = 复制文件（保留原文件）
    """
    uid_map = build_uid_map(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    all_nii = sorted(
        [str(p) for p in Path(src_dir).rglob("*.nii")] +
        [str(p) for p in Path(src_dir).rglob("*.nii.gz")]
    )
    print(f"找到 {len(all_nii)} 个 NIfTI 文件")

    copied = skipped_exists = not_in_list = 0
    for filepath in tqdm(all_nii, desc="整理中"):
        img_id = extract_image_id(filepath)
        if img_id is None or img_id not in uid_map:
            not_in_list += 1
            continue

        ptid, viscode = uid_map[img_id]
        ext = ".nii.gz" if filepath.endswith(".nii.gz") else ".nii"
        dst = os.path.join(out_dir, f"{ptid}_{viscode}{ext}")

        if os.path.exists(dst):
            skipped_exists += 1
            if move:
                os.remove(filepath)   # 已有目标文件，删除源文件
            continue

        if move:
            shutil.move(filepath, dst)
        else:
            shutil.copy2(filepath, dst)
        copied += 1

    action = "移动" if move else "复制"
    print(f"\n{action}: {copied}  |  已存在(跳过): {skipped_exists}  "
          f"|  不在下载清单: {not_in_list}")
    if move:
        print("提示：原文件已删除。若 ADNI/ 目录现在都是空子目录，可用 "
              "`rm -rf data/mri_downloaded/ADNI/` 清理。")


def organize_from_zip(zip_path: str, csv_path: str, out_dir: str) -> None:
    """
    流式从 zip 提取，一次只解压一个文件（~50 MB 临时空间），
    重命名后放到 out_dir/{PTID}_{VISCODE}.nii。
    支持断点续传：已存在的文件自动跳过。
    """
    uid_map = build_uid_map(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist()
                   if m.filename.endswith((".nii", ".nii.gz")) and not m.is_dir()]
        print(f"Zip 中找到 {len(members)} 个 NIfTI 文件")

        extracted = skipped_exists = not_in_list = 0
        for member in tqdm(members, desc="流式解压"):
            img_id = extract_image_id(member.filename)
            if img_id is None or img_id not in uid_map:
                not_in_list += 1
                continue

            ptid, viscode = uid_map[img_id]
            ext = ".nii.gz" if member.filename.endswith(".nii.gz") else ".nii"
            dst = os.path.join(out_dir, f"{ptid}_{viscode}{ext}")

            if os.path.exists(dst):
                skipped_exists += 1
                continue

            # 解压到临时目录，再移动到目标（临时占用 ~一个文件大小）
            with tempfile.TemporaryDirectory() as tmp:
                zf.extract(member, tmp)
                src = os.path.join(tmp, member.filename)
                shutil.move(src, dst)
            extracted += 1

    print(f"\n提取: {extracted}  |  已存在(跳过): {skipped_exists}  "
          f"|  不在下载清单: {not_in_list}")
    print(f"输出目录: {out_dir}")
    print(f"文件数:   {len(list(Path(out_dir).glob('*.nii*')))}")


def organize_by_date(src_dir: str, csv_path: str, out_dir: str,
                     move: bool = False) -> None:
    """
    按 PTID + 检查日期匹配，处理 Image ID 不在下载清单里的文件。

    匹配逻辑：
      CSV 的 EXAMDATE 列（YYYY-MM-DD）与文件路径中的日期目录匹配。
      同一 PTID+日期只保留第一个匹配文件。
    """
    dl = pd.read_csv(csv_path)
    dl["EXAMDATE"] = pd.to_datetime(dl["EXAMDATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    dl = dl.dropna(subset=["PTID", "VISCODE", "EXAMDATE"])
    # 构建 (PTID, EXAMDATE) → VISCODE 映射
    date_map = {}
    for _, row in dl.iterrows():
        key = (str(row["PTID"]), str(row["EXAMDATE"]))
        if key not in date_map:
            date_map[key] = str(row["VISCODE"])

    os.makedirs(out_dir, exist_ok=True)
    all_nii = sorted(
        [str(p) for p in Path(src_dir).rglob("*.nii")] +
        [str(p) for p in Path(src_dir).rglob("*.nii.gz")]
    )
    print(f"找到 {len(all_nii)} 个 NIfTI 文件（按日期匹配模式）")

    moved = skipped = no_match = already_exists = 0
    for filepath in tqdm(all_nii, desc="整理中"):
        ptid = re.search(r"(\d{3}_S_\d{4})", filepath)
        date = re.search(r"(\d{4}-\d{2}-\d{2})", filepath)
        if not ptid or not date:
            no_match += 1
            continue
        ptid, date = ptid.group(1), date.group(1)
        viscode = date_map.get((ptid, date))
        if viscode is None:
            no_match += 1
            continue

        ext      = ".nii.gz" if filepath.endswith(".nii.gz") else ".nii"
        out_name = f"{ptid}_{viscode}{ext}"
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path):
            already_exists += 1
            continue

        if move:
            shutil.move(filepath, out_path)
            moved += 1
        else:
            shutil.copy2(filepath, out_path)
            moved += 1

    print(f"\n移动: {moved}  |  已存在(跳过): {already_exists}  |  无法匹配: {no_match}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ADNI MRI 流式提取与整理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", help="子命令")

    # ── organize：整理已解压目录 ──────────────────────────────────────────────
    p1 = sub.add_parser("organize", help="整理已解压的 ADNI/ 目录")
    p1.add_argument("--src_dir", required=True,
                    help="ADNI 解压后的根目录（含 SubjectID 子目录）")
    p1.add_argument("--csv", default="data/mri_download_list.csv",
                    help="mri_download_list.csv 路径（含 PTID/VISCODE/IMAGEUID）")
    p1.add_argument("--out_dir", required=True,
                    help="输出目录（文件重命名为 PTID_VISCODE.nii）")
    p1.add_argument("--move", action="store_true",
                    help="移动文件（节省空间）而非复制")

    # ── organize_by_date：按日期匹配（Image ID 不在清单时的备用方案）─────────
    p3 = sub.add_parser("organize_by_date", help="按 PTID+日期匹配（Image ID 不在清单时使用）")
    p3.add_argument("--src_dir", required=True)
    p3.add_argument("--csv", default="data/mri_download_list.csv")
    p3.add_argument("--out_dir", required=True)
    p3.add_argument("--move", action="store_true", help="移动文件而非复制")

    # ── from_zip：流式解压 ────────────────────────────────────────────────────
    p2 = sub.add_parser("from_zip", help="流式从 zip 提取（最小临时磁盘占用）")
    p2.add_argument("--zip", required=True, help="ADNI zip 文件路径")
    p2.add_argument("--csv", default="data/mri_download_list.csv")
    p2.add_argument("--out_dir", required=True, help="输出目录")

    args = parser.parse_args()

    if args.cmd == "organize":
        organize_from_dir(args.src_dir, args.csv, args.out_dir, move=args.move)
    elif args.cmd == "from_zip":
        organize_from_zip(args.zip, args.csv, args.out_dir)
    elif args.cmd == "organize_by_date":
        organize_by_date(args.src_dir, args.csv, args.out_dir, move=args.move)
    else:
        parser.print_help()
