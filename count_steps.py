"""
统计各步骤 (t → t+1) 的样本数分布。
在 LMDP 目录下运行: python count_steps.py
"""
import pickle, collections
from config import Config

with open(Config.TAB_PROCESSED_PATH, "rb") as f:
    data = pickle.load(f)

all_ptids = list(data.keys())
total_subjects = len(all_ptids)

# 各序列长度分布
len_counter = collections.Counter()
# 各步骤预测对数
step_counter = collections.Counter()
# 各步骤中有MRI/PET可用的数量
step_mri = collections.Counter()
step_pet = collections.Counter()

for ptid in all_ptids:
    visits = data[ptid]["visits"][:Config.MAX_SEQ_LEN]
    T = len(visits)
    len_counter[T] += 1
    for t in range(T - 1):   # t→t+1 预测对
        step_counter[t] += 1
        if visits[t]["mri_path"] is not None:
            step_mri[t] += 1
        if visits[t]["pet_path"] is not None:
            step_pet[t] += 1

print(f"\n总受试者数: {total_subjects}")
print(f"\n序列长度分布（有多少次访视）:")
for l in sorted(len_counter):
    print(f"  {l} 次访视: {len_counter[l]} 人")

print(f"\n各步骤预测对数（全部受试者）:")
print(f"  {'步骤':>4}  {'预测对':>6}  {'MRI可用':>8}  {'PET可用':>8}  {'含义'}")
for t in sorted(step_counter):
    n = step_counter[t]
    print(f"  t={t}    {n:>6}    {step_mri[t]:>6}({step_mri[t]/n:.0%})  "
          f"{step_pet[t]:>6}({step_pet[t]/n:.0%})  "
          f"{'只有基线→预测第1次随访' if t==0 else f'有{t}次历史→预测第{t+1}次随访'}")

# filter_no_image 后的情况
has_any_img = [p for p in all_ptids
               if any(v["mri_path"] or v["pet_path"]
                      for v in data[p]["visits"])]
print(f"\n--filter_no_image 后: {len(has_any_img)} 人（删除 {total_subjects-len(has_any_img)} 人）")

step_c2 = collections.Counter()
for ptid in has_any_img:
    visits = data[ptid]["visits"][:Config.MAX_SEQ_LEN]
    for t in range(len(visits) - 1):
        step_c2[t] += 1

print(f"\n各步骤预测对数（filter_no_image 后）:")
for t in sorted(step_c2):
    print(f"  t={t}: {step_c2[t]} 对")

total_pairs = sum(step_c2.values())
print(f"\n总预测对数: {total_pairs}")
print(f"5折交叉验证下，每折验证集约: {total_pairs//5} 对")
print(f"（与你看到的 val 样本数对应）")
