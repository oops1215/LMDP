"""
从 ADNIMERGE2 R 包的 .rda 文件构建 ADNIMERGE.csv
=======================================================

合并以下文件（对应论文所需的全部特征）：
  DXSUM.rda       → 诊断（DX）
  PTDEMOG.rda     → 人口统计（年龄、性别、教育）
  APOERES.rda     → 遗传（APOE4）
  REGISTRY.rda    → 访视日期（EXAMDATE）
  UCSFFSX*.rda    → FreeSurfer 脑区体积（生物标志物 + ICV）

使用方法：
  python data/build_adnimerge_from_rda.py \\
      --rda_dir ~/LMDP/ADNIMERGE2/data \\
      --output  data/ADNIMERGE.csv

  # 先做列名诊断（不生成 CSV，仅打印找到/缺失的列）
  python data/build_adnimerge_from_rda.py --rda_dir ~/LMDP/ADNIMERGE2/data --inspect_only

依赖：pip install pyreadr pandas numpy
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ─── FreeSurfer 列名候选（按优先级排列，兼容多个 FS 版本）─────────────────────
# 对每个脑区，依次尝试列表中的列名，找到即用
FS_SINGLE_COLS = {
    "ICV": [
        "ST10CV", "ICV", "eTIV", "ETIV",
        "EstimatedTotalIntraCranialVol", "IntraCranialVol",
        "ST10SV", "BrainSegVol",
    ],
    "WholeBrain": [
        "ST133SV", "ST133CV", "BrainSegVol", "BrainSegVolNotVent",
        "WholeBrain", "TotalBrainVol",
    ],
}

FS_BILATERAL_COLS = {
    "Ventricles": {
        "left":  ["ST37SV", "ST37CV", "Left-Lateral-Ventricle",  "lLateralVentricle"],
        "right": ["ST96SV", "ST96CV", "Right-Lateral-Ventricle", "rLateralVentricle"],
        "extra_left":  ["ST30SV", "ST30CV"],   # inferior lateral ventricle
        "extra_right": ["ST89SV", "ST89CV"],
    },
    "Hippocampus": {
        "left":  ["ST29SV", "ST29CV", "Left-Hippocampus",  "lHippocampus"],
        "right": ["ST88SV", "ST88CV", "Right-Hippocampus", "rHippocampus"],
    },
    "Entorhinal": {
        "left":  ["ST24CV", "ST24TA", "ST24SA", "Left-Entorhinal",  "lEntorhinal"],
        "right": ["ST83CV", "ST83TA", "ST83SA", "Right-Entorhinal", "rEntorhinal"],
    },
    "Fusiform": {
        "left":  ["ST26CV", "ST26TA", "ST26SA", "Left-Fusiform",  "lFusiform"],
        "right": ["ST85CV", "ST85TA", "ST85SA", "Right-Fusiform", "rFusiform"],
    },
    "MidTemp": {
        "left":  ["ST40CV", "ST40TA", "ST40SA", "Left-Middle-Temporal",  "lMiddleTemporal"],
        "right": ["ST99CV", "ST99TA", "ST99SA", "Right-Middle-Temporal", "rMiddleTemporal"],
    },
}

# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def read_rda(rda_dir: str, filename: str) -> Optional[pd.DataFrame]:
    """用 pyreadr 读取 .rda 文件，返回 DataFrame（失败时返回 None）。"""
    try:
        import pyreadr
    except ImportError:
        print("[错误] 请先安装 pyreadr：pip install pyreadr")
        sys.exit(1)

    path = os.path.join(rda_dir, filename)
    if not os.path.exists(path):
        print(f"  [警告] 文件不存在：{path}")
        return None
    try:
        result = pyreadr.read_r(path)
        key = list(result.keys())[0]
        df = result[key]
        # 所有列名转大写，方便匹配
        df.columns = [c.upper() for c in df.columns]
        print(f"  读取 {filename}: {len(df)} 行 × {len(df.columns)} 列")
        return df
    except Exception as e:
        print(f"  [错误] 读取 {filename} 失败: {e}")
        return None


def find_col(df: pd.DataFrame, candidates: list, label: str) -> Optional[str]:
    """从候选列名列表中找到第一个存在于 df 的列名。"""
    for c in candidates:
        if c.upper() in df.columns:
            return c.upper()
    return None


def normalize_rid(df: pd.DataFrame) -> pd.DataFrame:
    """将 RID 列统一转换为 Int64（可空整数），避免 float64 vs object 的 merge 报错。"""
    if "RID" in df.columns:
        df = df.copy()
        df["RID"] = pd.to_numeric(df["RID"], errors="coerce").astype("Int64")
    return df


def try_get_bilateral(df: pd.DataFrame,
                       struct: str,
                       spec: dict) -> Optional[pd.Series]:
    """
    尝试从 df 中提取双侧体积之和。
    spec 包含 "left", "right" 候选列名列表，以及可选的 "extra_left", "extra_right"。
    """
    left_col  = find_col(df, spec["left"],  f"{struct}_left")
    right_col = find_col(df, spec["right"], f"{struct}_right")
    if left_col is None or right_col is None:
        return None

    result = pd.to_numeric(df[left_col], errors="coerce") + \
             pd.to_numeric(df[right_col], errors="coerce")

    # 侧脑室：加入下侧脑室（inferior lateral ventricle）
    if "extra_left" in spec and "extra_right" in spec:
        el = find_col(df, spec["extra_left"],  f"{struct}_extra_left")
        er = find_col(df, spec["extra_right"], f"{struct}_extra_right")
        if el and er:
            result += pd.to_numeric(df[el], errors="coerce").fillna(0) + \
                      pd.to_numeric(df[er], errors="coerce").fillna(0)
    return result


def standardize_viscode(vc: str) -> str:
    """将原始 VISCODE 标准化为 bl/m06/m12/... 格式。"""
    if not isinstance(vc, str):
        return ""
    vc = vc.strip().lower()
    # 基线访视的各种名称
    if vc in ("bl", "sc", "scmri", "init", "nv", "v1", "bas",
              "m0", "00m", "baseline", "adni screening"):
        return "bl"
    # 月份代码：m06, m12, m24 等，也处理 6m/12m 格式
    import re
    m = re.match(r"m?(\d+)m?$", vc)
    if m:
        months = int(m.group(1))
        return f"m{months:02d}" if months < 10 else f"m{months}"
    return vc


# ─── 步骤1：诊断（DXSUM）──────────────────────────────────────────────────────

def load_dxsum(rda_dir: str) -> pd.DataFrame:
    df = read_rda(rda_dir, "DXSUM.rda")
    if df is None:
        raise FileNotFoundError("DXSUM.rda 是必需文件，请确认路径正确")

    # 确认关键列存在
    assert "RID" in df.columns, f"DXSUM 中没有 RID 列，实际列: {list(df.columns)[:20]}"

    # 标准化访视代码（优先 VISCODE2）
    vc_col = "VISCODE2" if "VISCODE2" in df.columns else "VISCODE"
    df["VISCODE_STD"] = df[vc_col].apply(standardize_viscode)

    # 解析诊断
    dx_map_curren  = {1: "CN", 2: "MCI", 3: "Dementia"}
    # DXCHANGE: 1=Stable NL, 2=Stable MCI, 3=Stable Dem,
    #           4=Conv NL→MCI, 5=Conv MCI→Dem, 6=Conv NL→Dem,
    #           7=Rev MCI→NL, 8=Rev Dem→MCI, 9=Rev Dem→NL
    dx_map_change  = {
        1: "CN", 2: "MCI", 3: "Dementia",
        4: "MCI", 5: "Dementia", 6: "Dementia",
        7: "CN",  8: "MCI",  9: "CN",
    }

    # 直接字符串诊断列（部分 ADNIMERGE2 版本已有处理好的列）
    dx_str_map = {
        "nl": "CN", "cn": "CN", "normal": "CN",
        "mci": "MCI", "emci": "MCI", "lmci": "MCI",
        "ad": "Dementia", "dem": "Dementia", "dementia": "Dementia",
    }

    def _to_int(val):
        """安全地将 val 转为 int，处理 1.0 / '1' / '1.0' 等各种形式。"""
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    def parse_dx(row):
        # 优先：直接字符串诊断列（ADNI3/ADNIMERGE2 新版）
        for col in ["DIAGNOSIS", "DXMDES", "DX"]:
            if col in df.columns:
                val = str(row.get(col, "")).strip().lower()
                if val in dx_str_map:
                    return dx_str_map[val]
        # ADNI-GO/2/3：DXCHANGE（数值编码）
        if "DXCHANGE" in df.columns:
            v = _to_int(row.get("DXCHANGE"))
            if v is not None:
                r = dx_map_change.get(v, "")
                if r:
                    return r
        # ADNI1：DXCURREN
        if "DXCURREN" in df.columns:
            v = _to_int(row.get("DXCURREN"))
            if v is not None:
                r = dx_map_curren.get(v, "")
                if r:
                    return r
        return ""

    df["DX"] = df.apply(parse_dx, axis=1)
    df = df[df["DX"] != ""]  # 去掉没有有效诊断的行

    # 取 EXAMDATE（不同版本列名可能不同）
    exam_col = find_col(df, ["EXAMDATE", "USERDATE", "USERDATE2"], "EXAMDATE")
    if exam_col:
        df["EXAMDATE"] = pd.to_datetime(df[exam_col], errors="coerce")

    out = df[["RID", "VISCODE_STD", "DX"] +
             (["EXAMDATE"] if "EXAMDATE" in df.columns else [])].copy()
    out = out.rename(columns={"VISCODE_STD": "VISCODE"})
    # 去重（同一 RID + VISCODE 保留第一条）
    out = out.sort_values("RID").drop_duplicates(subset=["RID", "VISCODE"], keep="first")
    out = normalize_rid(out)

    if len(out) == 0:
        # 打印实际列名帮助调试
        dx_cols = [c for c in df.columns if any(
            k in c for k in ["DX", "DIAG", "CURREN", "CHANGE"])]
        print(f"  [诊断] 检测到的相关列: {dx_cols}")
        sample_vals = {}
        for c in dx_cols[:3]:
            sample_vals[c] = df[c].dropna().unique()[:5].tolist()
        print(f"  [诊断] 样例值: {sample_vals}")

    print(f"  → 有效诊断记录: {len(out)} 条，受试者: {out['RID'].nunique()} 人")
    return out


# ─── 步骤2：访视日期（REGISTRY）──────────────────────────────────────────────

def load_registry(rda_dir: str) -> pd.DataFrame:
    df = read_rda(rda_dir, "REGISTRY.rda")
    if df is None:
        return pd.DataFrame(columns=["RID", "VISCODE", "EXAMDATE"])

    vc_col = "VISCODE2" if "VISCODE2" in df.columns else "VISCODE"
    df["VISCODE_STD"] = df[vc_col].apply(standardize_viscode)
    df["EXAMDATE_REG"] = pd.to_datetime(
        df.get("EXAMDATE", df.get("USERDATE")), errors="coerce")

    out = df[["RID", "VISCODE_STD", "EXAMDATE_REG"]].copy()
    out = out.rename(columns={"VISCODE_STD": "VISCODE"})
    out = out.drop_duplicates(subset=["RID", "VISCODE"], keep="first")
    out = normalize_rid(out)
    return out


# ─── 步骤3：人口统计（PTDEMOG）────────────────────────────────────────────────

def load_ptdemog(rda_dir: str) -> pd.DataFrame:
    df = read_rda(rda_dir, "PTDEMOG.rda")
    if df is None:
        return pd.DataFrame(columns=["RID", "PTID", "PTGENDER", "PTEDUCAT",
                                      "PTDOB_YEAR", "PTDOB_MONTH"])

    # PTID（格式 XXX_S_XXXX）
    ptid_col = find_col(df, ["PTID"], "PTID")

    # 性别（1=Male, 2=Female）
    gender_col = find_col(df, ["PTGENDER"], "PTGENDER")

    # 教育
    edu_col = find_col(df, ["PTEDUCAT"], "PTEDUCAT")

    # 出生年月（用于计算年龄）
    dob_year_col  = find_col(df, [
        "PTDOBYY", "BIRTHYR", "BIRTHYEAR", "DOBYR", "PTYEAR",
        "YOB", "YEAR_OF_BIRTH", "BIRTH_YEAR", "PTBIRTHYEAR",
    ], "DOB_YEAR")
    dob_month_col = find_col(df, [
        "PTDOBMM", "BIRTHMO", "BIRTHMONTH", "DOBMM", "PTMONTH",
        "MOB", "MONTH_OF_BIRTH", "BIRTH_MONTH",
    ], "DOB_MONTH")

    # 直接年龄列（基线年龄，部分版本直接提供）
    age_direct_col = find_col(df, ["AGE", "PTAGE", "AGECONS", "AGE_CONSENT",
                                   "PTAGEYRS", "AGEYRBL"], "AGE")

    if dob_year_col is None and age_direct_col is None:
        # 打印所有列名帮助调试
        print(f"  [警告] PTDEMOG 中未找到出生年份或年龄列，AGE 将缺失")
        age_related = [c for c in df.columns
                       if any(k in c for k in ["DOB", "BIRTH", "AGE", "YR", "YEAR"])]
        print(f"         相关列候选: {age_related[:20]}")

    cols = ["RID"]
    rename = {}
    for c, name in [
        (ptid_col,      "PTID"),
        (gender_col,    "PTGENDER"),
        (edu_col,       "PTEDUCAT"),
        (dob_year_col,  "PTDOB_YEAR"),
        (dob_month_col, "PTDOB_MONTH"),
        (age_direct_col,"AGE_DIRECT"),   # 直接年龄（基线）
    ]:
        if c:
            cols.append(c)
            rename[c] = name

    out = df[cols].copy().rename(columns=rename)
    # PTDEMOG 是一次性的（每人一条基线记录），去重
    out = out.drop_duplicates(subset=["RID"], keep="first")
    out = normalize_rid(out)
    return out


# ─── 步骤4：APOE4（APOERES）──────────────────────────────────────────────────

def load_apoeres(rda_dir: str) -> pd.DataFrame:
    df = read_rda(rda_dir, "APOERES.rda")
    if df is None:
        return pd.DataFrame(columns=["RID", "APOE4"])

    # 先检查是否已有直接的 APOE4 计数列
    apoe4_direct = find_col(df, ["APOE4", "APOE4NUM", "APOE_E4"], "APOE4")
    if apoe4_direct:
        df["APOE4"] = pd.to_numeric(df[apoe4_direct], errors="coerce")
        out = normalize_rid(df[["RID", "APOE4"]].drop_duplicates(subset=["RID"], keep="first"))
        return out

    # GENOTYPE 列（格式如 "3/4"、"4/4"、"2/3"）→ 计算 ε4 等位基因数
    genotype_col = find_col(df, ["GENOTYPE", "APOE_GENOTYPE", "APOETYPE"], "GENOTYPE")
    if genotype_col:
        def count_e4(gt):
            if not isinstance(gt, str):
                return np.nan
            alleles = gt.strip().replace(" ", "").split("/")
            return sum(1 for a in alleles if a == "4")
        df["APOE4"] = df[genotype_col].apply(count_e4)
        out = normalize_rid(df[["RID", "APOE4"]].drop_duplicates(subset=["RID"], keep="first"))
        return out

    # 从两个等位基因列计算（APGEN1/APGEN2 = allele codes, 4 = ε4）
    g1 = find_col(df, ["APGEN1", "ALLELE1", "APOE_ALLELE1", "GENE1"], "APGEN1")
    g2 = find_col(df, ["APGEN2", "ALLELE2", "APOE_ALLELE2", "GENE2"], "APGEN2")
    if g1 is None or g2 is None:
        print("  [警告] APOERES 中未找到等位基因列，APOE4 将缺失")
        print(f"         实际列名: {list(df.columns)}")
        empty = pd.DataFrame({"RID": pd.array([], dtype="Int64"), "APOE4": []})
        return empty

    df["APOE4"] = (pd.to_numeric(df[g1], errors="coerce").eq(4).astype(int) +
                   pd.to_numeric(df[g2], errors="coerce").eq(4).astype(int))
    out = normalize_rid(df[["RID", "APOE4"]].drop_duplicates(subset=["RID"], keep="first"))
    return out


# ─── 步骤5：FreeSurfer 脑区体积 ───────────────────────────────────────────────

def load_freesurfer_single(rda_dir: str, filename: str) -> Optional[pd.DataFrame]:
    """
    读取一个 UCSFFSX*.rda 文件，提取标准化列名的体积数据。
    仅保留 OVERALLQC == 'Pass'（或 1）的记录。
    """
    df = read_rda(rda_dir, filename)
    if df is None:
        return None

    vc_col = "VISCODE2" if "VISCODE2" in df.columns else "VISCODE"
    df["VISCODE_STD"] = df[vc_col].apply(standardize_viscode)

    # QC 过滤（保留质量合格的扫描）
    qc_col = find_col(df, ["OVERALLQC"], "OVERALLQC")
    if qc_col:
        df = df[df[qc_col].astype(str).str.strip().str.upper().isin(
            ["PASS", "1", "1.0", "TRUE"])]

    out = df[["RID", "VISCODE_STD"]].copy()
    out = out.rename(columns={"VISCODE_STD": "VISCODE"})

    # ICV（单列）
    for var, candidates in FS_SINGLE_COLS.items():
        col = find_col(df, candidates, var)
        out[var] = pd.to_numeric(df[col], errors="coerce") if col else np.nan

    # 双侧体积之和
    for var, spec in FS_BILATERAL_COLS.items():
        series = try_get_bilateral(df, var, spec)
        out[var] = series if series is not None else np.nan

    out = out.drop_duplicates(subset=["RID", "VISCODE"], keep="first")
    out = normalize_rid(out)
    n_valid = out[["Hippocampus", "Ventricles", "ICV"]].notna().all(axis=1).sum()
    print(f"    {filename}: {len(out)} 行，含三项指标 {n_valid} 条")
    return out


def load_all_freesurfer(rda_dir: str) -> pd.DataFrame:
    """
    依次读取各相位的 FreeSurfer 文件并合并（后加载的文件不覆盖已有数据）。
    优先级（从低到高）：UCSFFSX → UCSFFSX51ALL → UCSFFSX51_ADNI1_3T
                         → UCSFFSX6 → UCSFFSX7
    """
    print("\n  读取 FreeSurfer 文件 ...")
    # 后面的文件优先级更高（覆盖同一 RID+VISCODE 的之前结果）
    fs_files = [
        "UCSFFSX.rda",
        "UCSFFSX51ALL.rda",
        "UCSFFSX51_ADNI1_3T.rda",
        "UCSFFSX6.rda",
        "UCSFFSX7.rda",
    ]
    frames = []
    for fname in fs_files:
        sub = load_freesurfer_single(rda_dir, fname)
        if sub is not None:
            frames.append(sub)

    if not frames:
        print("  [警告] 未找到任何 FreeSurfer 文件，体积特征将全部缺失")
        return pd.DataFrame(columns=["RID", "VISCODE"] +
                             list(FS_SINGLE_COLS) + list(FS_BILATERAL_COLS))

    # 合并：后加载的优先（覆盖同一 RID+VISCODE）
    merged = frames[0]
    for f in frames[1:]:
        merged = pd.concat([f, merged], ignore_index=True)  # f 在前 → 优先保留
    merged = merged.drop_duplicates(subset=["RID", "VISCODE"], keep="first")
    print(f"  → FreeSurfer 合并后: {len(merged)} 条，受试者: {merged['RID'].nunique()} 人")
    return merged


# ─── 步骤6：计算年龄 ────────────────────────────────────────────────────────

def compute_age(demo_df: pd.DataFrame,
                dxsum_df: pd.DataFrame) -> pd.DataFrame:
    """
    利用 PTDEMOG 的出生年月（或直接年龄列）和 DXSUM 的 EXAMDATE 计算各访视的年龄。
    优先级：出生年份+EXAMDATE > 直接年龄列（仅基线准确，其他访视按时间偏移估算）
    """
    # 方案 A：用出生年月 + EXAMDATE 精确计算每次访视的年龄
    if "PTDOB_YEAR" in demo_df.columns and "EXAMDATE" in dxsum_df.columns:
        keep_cols = ["RID", "PTDOB_YEAR"] + \
                    (["PTDOB_MONTH"] if "PTDOB_MONTH" in demo_df.columns else [])
        dob = demo_df[keep_cols].copy()
        if "PTDOB_MONTH" not in dob.columns:
            dob["PTDOB_MONTH"] = 7
        dob["PTDOB_MONTH"] = pd.to_numeric(dob["PTDOB_MONTH"],
                                            errors="coerce").fillna(7).astype(int)
        # 转为整数再组字符串（避免 float → "1935.0-7-15" 无法解析的问题）
        dob["PTDOB_YEAR"] = pd.to_numeric(dob["PTDOB_YEAR"], errors="coerce")
        year_s  = dob["PTDOB_YEAR"].apply(
            lambda x: str(int(x)) if pd.notna(x) else np.nan)
        month_s = dob["PTDOB_MONTH"].apply(lambda x: f"{int(x):02d}")
        dob["DOB_DATE"] = pd.to_datetime(
            year_s + "-" + month_s + "-15", errors="coerce"
        )
        merged = dxsum_df.merge(dob[["RID", "DOB_DATE"]], on="RID", how="left")
        merged["AGE"] = (pd.to_datetime(merged["EXAMDATE"], errors="coerce") -
                         merged["DOB_DATE"]).dt.days / 365.25
        merged = merged.drop(columns=["DOB_DATE"])
        return merged

    # 方案 B：直接年龄列（仅为基线年龄，非精确）
    if "AGE_DIRECT" in demo_df.columns:
        age_map = demo_df.set_index("RID")["AGE_DIRECT"]
        result = dxsum_df.copy()
        result["AGE"] = result["RID"].map(age_map)
        result["AGE"] = pd.to_numeric(result["AGE"], errors="coerce")
        print("  [年龄] 使用直接年龄列（基线值，其他访视未按时间偏移）")
        return result

    return dxsum_df


# ─── 步骤7：列名诊断模式 ────────────────────────────────────────────────────

def inspect_columns(rda_dir: str) -> None:
    """打印各 .rda 文件中与论文相关的列，帮助调试列名不匹配问题。"""
    inspect_targets = {
        "DXSUM.rda":       ["DXCHANGE", "DXCURREN", "VISCODE", "VISCODE2", "EXAMDATE", "RID"],
        "PTDEMOG.rda":     ["PTGENDER", "PTEDUCAT", "PTDOBYY", "PTDOBMM", "PTID", "RID",
                             "AGE", "PTAGE", "AGECONS", "BIRTHYR", "BIRTHYEAR"],
        "APOERES.rda":     ["APGEN1", "APGEN2", "RID"],
        "REGISTRY.rda":    ["EXAMDATE", "VISCODE", "VISCODE2", "RID"],
        "UCSFFSX51ALL.rda":["ST10CV", "ST29SV", "ST88SV", "ST37SV", "ST96SV",
                             "ST24CV", "ST83CV", "ST26CV", "ST85CV", "ST40CV", "ST99CV",
                             "ST133SV", "OVERALLQC", "RID", "VISCODE2"],
        "UCSFFSX.rda":     ["ST10CV", "ST29SV", "ST88SV", "OVERALLQC", "RID"],
        "UCSFFSX6.rda":    ["ST10CV", "ST29SV", "ST88SV", "OVERALLQC", "RID"],
        "UCSFFSX7.rda":    ["ST10CV", "ST29SV", "ST88SV", "OVERALLQC", "RID"],
    }

    for fname, expected_cols in inspect_targets.items():
        df = read_rda(rda_dir, fname)
        if df is None:
            continue
        print(f"\n  {fname} ({len(df)} 行):")
        found, missing = [], []
        for c in expected_cols:
            (found if c.upper() in df.columns else missing).append(c)
        print(f"    ✓ 找到: {found}")
        print(f"    ✗ 缺失: {missing}")
        # 打印所有含关键词的列（帮助找到替代名）
        keywords = ["HIPPO", "VENT", "ENTORH", "FUSIFORM", "TEMP",
                    "ICV", "BRAIN", "APOE", "GENDER", "EDUCAT", "DXCH"]
        related = [c for c in df.columns
                   if any(k in c.upper() for k in keywords)]
        if related:
            print(f"    相关列: {related[:30]}")


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def build_adnimerge(rda_dir: str,
                    output_path: str,
                    inspect_only: bool = False) -> None:
    rda_dir = os.path.expanduser(rda_dir)
    if not os.path.isdir(rda_dir):
        print(f"[错误] 目录不存在: {rda_dir}")
        sys.exit(1)

    print(f"\nADNIMERGE2 数据目录: {rda_dir}")

    if inspect_only:
        print("\n=== 列名诊断模式 ===")
        inspect_columns(rda_dir)
        return

    print("\n=== 开始构建 ADNIMERGE.csv ===")

    # ── 读取各组件 ────────────────────────────────────────────────────────────
    print("\n1. 读取诊断数据 (DXSUM) ...")
    dxsum = load_dxsum(rda_dir)

    print("\n2. 读取访视日期 (REGISTRY) ...")
    registry = load_registry(rda_dir)

    print("\n3. 读取人口统计 (PTDEMOG) ...")
    demo = load_ptdemog(rda_dir)

    print("\n4. 读取 APOE4 基因型 (APOERES) ...")
    apoe = load_apoeres(rda_dir)

    print("\n5. 读取 FreeSurfer 脑区体积 ...")
    fs = load_all_freesurfer(rda_dir)

    # ── 合并 ──────────────────────────────────────────────────────────────────
    print("\n6. 合并所有数据 ...")

    # DXSUM 补充来自 REGISTRY 的 EXAMDATE（DXSUM 自带 EXAMDATE 优先）
    if "EXAMDATE" not in dxsum.columns and "EXAMDATE_REG" in registry.columns:
        dxsum = dxsum.merge(
            registry.rename(columns={"EXAMDATE_REG": "EXAMDATE"}),
            on=["RID", "VISCODE"], how="left",
        )
    elif "EXAMDATE_REG" in registry.columns:
        dxsum = dxsum.merge(
            registry[["RID", "VISCODE", "EXAMDATE_REG"]],
            on=["RID", "VISCODE"], how="left",
        )
        # 用 REGISTRY 日期填充 DXSUM 缺失的 EXAMDATE
        dxsum["EXAMDATE"] = dxsum["EXAMDATE"].fillna(dxsum["EXAMDATE_REG"])
        dxsum = dxsum.drop(columns=["EXAMDATE_REG"])

    # 加入人口统计（基线记录，按 RID join）
    merged = dxsum.merge(demo, on="RID", how="left")

    # 加入 APOE4
    merged = merged.merge(apoe, on="RID", how="left")

    # 计算年龄
    merged = compute_age(demo, merged)

    # 加入 FreeSurfer 体积
    merged = merged.merge(fs, on=["RID", "VISCODE"], how="left")

    # ── 生成 PTID（若 PTDEMOG 没有提供）────────────────────────────────────
    if "PTID" not in merged.columns:
        # PTID 格式：{SITEID:03d}_S_{RID:04d}
        # SITEID 在 DXSUM 或 demo 中
        site_col = find_col(merged, ["SITEID"], "SITEID")
        if site_col:
            merged["PTID"] = (
                merged[site_col].astype(str).str.zfill(3) + "_S_" +
                merged["RID"].astype(str).str.zfill(4)
            )
        else:
            merged["PTID"] = "000_S_" + merged["RID"].astype(str).str.zfill(4)

    # ── 将性别数字码转为字符串 ───────────────────────────────────────────────
    if "PTGENDER" in merged.columns:
        merged["PTGENDER"] = merged["PTGENDER"].map(
            {1: "Male", 2: "Female", "1": "Male", "2": "Female",
             1.0: "Male", 2.0: "Female"}
        ).fillna(merged["PTGENDER"])

    # ── 基线诊断 (DX_bl) ────────────────────────────────────────────────────
    bl_dx = (merged[merged["VISCODE"] == "bl"][["RID", "DX"]]
             .drop_duplicates("RID").rename(columns={"DX": "DX_bl"}))
    merged = merged.merge(bl_dx, on="RID", how="left")

    # ── EXAMDATE 格式化 ─────────────────────────────────────────────────────
    if "EXAMDATE" in merged.columns:
        merged["EXAMDATE"] = pd.to_datetime(merged["EXAMDATE"],
                                             errors="coerce").dt.strftime("%Y-%m-%d")

    # ── 最终列顺序（与 ADNIMERGE.csv 兼容）──────────────────────────────────
    final_cols = [
        "PTID", "RID", "VISCODE", "EXAMDATE",
        "DX", "DX_bl",
        "AGE", "PTGENDER", "PTEDUCAT",
        "APOE4",
        "Ventricles", "Hippocampus", "WholeBrain",
        "Entorhinal", "Fusiform", "MidTemp",
        "ICV",
    ]
    for c in final_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged[final_cols + [c for c in merged.columns if c not in final_cols]]

    # ── 保存 ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    merged.to_csv(output_path, index=False)

    # ── 统计报告 ─────────────────────────────────────────────────────────────
    print(f"\n=== 构建完成 ===")
    print(f"  总记录数    : {len(merged)}")
    print(f"  受试者数    : {merged['RID'].nunique()}")
    print(f"  访视分布:\n{merged['VISCODE'].value_counts().to_string()}")
    print(f"  诊断分布:\n{merged['DX'].value_counts().to_string()}")
    missing_pct = merged[["Hippocampus", "Ventricles", "ICV", "AGE", "APOE4"]].isna().mean() * 100
    print(f"\n  缺失率:")
    for col, pct in missing_pct.items():
        print(f"    {col:15s}: {pct:.1f}%")
    print(f"\n  保存到: {output_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从 ADNIMERGE2 .rda 文件构建 ADNIMERGE.csv")
    parser.add_argument("--rda_dir",      required=True,
                        help="ADNIMERGE2 的 data/ 目录路径，如 ~/LMDP/ADNIMERGE2/data")
    parser.add_argument("--output",       default="data/ADNIMERGE.csv",
                        help="输出 CSV 路径（默认 data/ADNIMERGE.csv）")
    parser.add_argument("--inspect_only", action="store_true",
                        help="仅打印列名诊断，不生成 CSV")
    args = parser.parse_args()

    build_adnimerge(args.rda_dir, args.output, args.inspect_only)
