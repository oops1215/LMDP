"""
端到端流水线：从 ADNIMERGE.csv + 原始图像到训练完成的 LMDP-Net

使用方法：
  # 完整流水线（需要下载好数据）
  python run_pipeline.py

  # 仅预处理表格数据（跳过图像）
  python run_pipeline.py --skip_imaging

  # 仅训练（数据已准备好）
  python run_pipeline.py --skip_preprocess

  # 不使用图像（只用表格数据，快速验证流程）
  python run_pipeline.py --no_images --skip_imaging
"""

import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def run(cmd: str, desc: str = "") -> int:
    if desc:
        print(f"\n{'─'*60}")
        print(f"  {desc}")
        print(f"{'─'*60}")
    print(f"  $ {cmd}")
    ret = subprocess.run(cmd, shell=True, cwd=ROOT).returncode
    if ret != 0:
        print(f"  [错误] 命令退出码 {ret}")
    return ret


def check_file(path: str, name: str) -> bool:
    if not os.path.exists(path):
        print(f"  [!] 找不到 {name}: {path}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="LMDP-Net 端到端流水线")
    parser.add_argument("--rda_dir",        default="",
                        help="ADNIMERGE2 data/ 目录（有 .rda 文件时自动构建 ADNIMERGE.csv）")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="跳过预处理步骤（数据已准备好）")
    parser.add_argument("--skip_imaging",    action="store_true",
                        help="跳过图像预处理（使用纯表格模式）")
    parser.add_argument("--no_images",       action="store_true",
                        help="训练时不加载图像特征")
    parser.add_argument("--fold",            type=int, default=-1,
                        help="-1=全部5折，否则指定某折")
    parser.add_argument("--transform",       default="Affine",
                        choices=["Affine", "SyN"],
                        help="图像配准方法：Affine(快) 或 SyN(精度高)")
    args = parser.parse_args()

    from config import Config

    print("=" * 60)
    print("  LMDP-Net 复现流水线")
    print("=" * 60)

    # ── 步骤 0：从 .rda 文件构建 ADNIMERGE.csv（若尚未存在）──────────────────
    if not args.skip_preprocess and not os.path.exists(Config.ADNIMERGE_PATH):
        rda_dir = args.rda_dir or os.path.expanduser("~/LMDP/ADNIMERGE2/data")
        if os.path.isdir(rda_dir) and any(
                f.endswith(".rda") for f in os.listdir(rda_dir)):
            ret = run(
                f"python {os.path.join('data', 'build_adnimerge_from_rda.py')} "
                f"--rda_dir \"{rda_dir}\" --output {Config.ADNIMERGE_PATH}",
                "步骤 0/3：从 .rda 文件构建 ADNIMERGE.csv"
            )
            if ret != 0:
                sys.exit(1)
        else:
            print(f"\n[错误] 找不到 {Config.ADNIMERGE_PATH}，也找不到 .rda 目录")
            print("  方案 A：提供 --rda_dir ~/LMDP/ADNIMERGE2/data")
            print("  方案 B：手动下载 ADNIMERGE.csv 到 data/ADNIMERGE.csv")
            sys.exit(1)

    # ── 步骤 1：表格数据预处理 ───────────────────────────────────────────────
    if not args.skip_preprocess:
        ret = run(
            f"python {os.path.join('data', 'preprocess_tabular.py')}",
            "步骤 1/3：预处理 ADNIMERGE.csv 表格数据"
        )
        if ret != 0:
            sys.exit(1)
    else:
        print("\n[跳过] 表格数据预处理")
        if not check_file(Config.TAB_PROCESSED_PATH, "tabular_processed.pkl"):
            sys.exit(1)

    # ── 步骤 2：图像预处理 ───────────────────────────────────────────────────
    if not args.skip_imaging and not args.no_images:
        mri_count = len([f for f in os.listdir(Config.MRI_RAW_DIR)
                         if f.endswith((".nii", ".nii.gz"))]) \
            if os.path.exists(Config.MRI_RAW_DIR) else 0
        pet_count = len([f for f in os.listdir(Config.PET_RAW_DIR)
                         if f.endswith((".nii", ".nii.gz"))]) \
            if os.path.exists(Config.PET_RAW_DIR) else 0

        if mri_count == 0 and pet_count == 0:
            print("\n[警告] 未找到原始图像文件。")
            print("  请先下载 ADNI MRI/PET 图像（参见 data/download_adni.py 中的说明）")
            print("  或使用 --no_images 跳过图像特征")
        else:
            print(f"\n  找到 MRI={mri_count} 个，PET={pet_count} 个原始图像")
            ret = run(
                f"python {os.path.join('data', 'preprocess_imaging.py')} "
                f"preprocess --transform {args.transform}",
                "步骤 2/3：图像配准与预处理"
            )
            if ret != 0:
                print("  [警告] 图像预处理失败，尝试以无图像模式继续...")
                args.no_images = True
    else:
        print("\n[跳过] 图像预处理")

    # ── 步骤 3：训练 ────────────────────────────────────────────────────────
    train_cmd = f"python train.py"
    if args.no_images:
        train_cmd += " --no_images"
    if args.fold >= 0:
        train_cmd += f" --fold {args.fold}"

    ret = run(
        train_cmd,
        f"步骤 3/3：训练 LMDP-Net（{'无图像模式' if args.no_images else '多模态模式'}）"
    )

    if ret == 0:
        print("\n✓ 流水线完成！模型权重保存在 checkpoints/ 目录")
    else:
        print("\n✗ 训练失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
