"""
gen_concept_json.py
--------------------
生成 concept_train_ctm_v1.json（9600 × 177）

方法（原始思想）：
  1. 找到文本的概率最大主题 t_text
  2. 找到形容词的概率最大主题 t_adj
  3. 用 Σ 协方差矩阵中 t_text 和 t_adj 之间的相关系数
     作为该文本和该形容词之间的相关程度

适配说明：
  - 当前使用 sklearn LDA 训练，Σ 是 10×10 完整的θ协方差矩阵
  - 被移除的 849 个全零样本，concept 设为均匀分布 1/N_ADJ
  - 输出维度：9600 × 177

用法：
  python scripts/gen_concept_json.py
"""

import json
import csv
import os
import numpy as np
import sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path("d:/CTM")
MODEL_DIR = BASE_DIR / "output" / "ctm_model"
PREP_DIR = BASE_DIR / "output" / "preprocessing"
DATA_DIR = BASE_DIR / "data" / "raw" / "TOXICN"
OUTPUT_DIR = BASE_DIR / "output" / "experiments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Step 1: 读取所有数据
# ============================================================
print("=" * 60)
print("Step 1: 读取数据")
print("=" * 60)

# 1a. 读取 train.json（原始标签）
print("读取 train.json ...")
with open(DATA_DIR / "train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
print(f"  原始样本数: {len(train_data)}")

# 1b. 读取被移除的文档
print("读取 removed_docs.csv ...")
removed_sample_ids = set()
removed_path = PREP_DIR / "removed_docs.csv"
if removed_path.exists():
    with open(removed_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["doc_type"] == "sample":
                removed_sample_ids.add(row["doc_id"])
print(f"  被移除的样本数: {len(removed_sample_ids)}")

# 1c. 读取 theta_sample.csv
print("读取 theta_sample.csv ...")
sample_ids = []
theta_sample = []
with open(MODEL_DIR / "theta_sample.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    K = len(header) - 1
    topic_names = header[1:]
    for row in reader:
        sample_ids.append(row[0])
        theta_sample.append([float(x) for x in row[1:]])
theta_sample = np.array(theta_sample)
N_SAMPLE = len(sample_ids)
print(f"  θ_sample: {theta_sample.shape}, K={K}")

# 1d. 读取 theta_adj.csv
print("读取 theta_adj.csv ...")
adj_ids = []
theta_adj = []
with open(MODEL_DIR / "theta_adj.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        adj_ids.append(row[0])
        theta_adj.append([float(x) for x in row[1:]])
theta_adj = np.array(theta_adj)
N_ADJ = len(adj_ids)
print(f"  θ_adj: {theta_adj.shape}")

# 1e. 读取形容词中文名
print("读取形容词词典 ...")
adj_path = BASE_DIR / "data" / "raw" / "adjective" / "toxic_adjectives_v1.csv"
adj_name_list = []
with open(adj_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        adj_name_list.append(row["chinese"])
print(f"  形容词数: {len(adj_name_list)}")

# 1f. 读取 Σ 协方差矩阵并计算相关系数矩阵
print("读取 sigma_matrix.csv 并计算相关系数矩阵 ...")
sigma_rows = []
sigma_topic_names = []
with open(MODEL_DIR / "sigma_matrix.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    sigma_topic_names = header[1:]
    for row in reader:
        sigma_rows.append([float(x) for x in row[1:]])
sigma = np.array(sigma_rows)

print(f"  Σ 维度: {sigma.shape[0]}×{sigma.shape[1]}")
print(f"  Σ 包含的主题: {sigma_topic_names}")

# 计算相关系数矩阵
# 当前使用 LDA，Σ 是完整的 K×K 协方差矩阵（不是 R CTM 的 (K-1)×(K-1)）
diag = np.sqrt(np.abs(np.diag(sigma)))
diag[diag == 0] = 1e-10
corr_matrix = sigma / np.outer(diag, diag)
np.fill_diagonal(corr_matrix, 1.0)

# 裁剪到 [-1, 1]（数值精度可能超出范围）
corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

print(f"  相关系数矩阵维度: {corr_matrix.shape}")

# 构建主题名 → 相关系数矩阵索引 的映射
topic_to_corr_idx = {}
for i, name in enumerate(sigma_topic_names):
    topic_to_corr_idx[name] = i

# 1g. 读取元数据获取文本
print("读取元数据获取文本 ...")
sample_texts = {}
with open(PREP_DIR / "dtm_metadata.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["doc_type"] == "sample":
            sample_texts[row["doc_id"]] = row["source"]
print(f"  样本文本数: {len(sample_texts)}")

# ============================================================
# Step 2: 计算每个样本和每个形容词的主导主题
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 确定主导主题")
print("=" * 60)

# 样本的主导主题（0-indexed）
sample_dominant = theta_sample.argmax(axis=1)
print(f"  样本主导主题分布:")
sample_topic_counter = Counter(sample_dominant)
for t in sorted(sample_topic_counter.keys()):
    print(f"    Topic{t+1}: {sample_topic_counter[t]} ({100*sample_topic_counter[t]/N_SAMPLE:.1f}%)")

# 形容词的主导主题
adj_dominant = theta_adj.argmax(axis=1)
print(f"  形容词主导主题分布:")
adj_topic_counter = Counter(adj_dominant)
for t in sorted(adj_topic_counter.keys()):
    print(f"    Topic{t+1}: {adj_topic_counter[t]} ({100*adj_topic_counter[t]/N_ADJ:.1f}%)")

# ============================================================
# Step 3: 构建主题间关联度查找表
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 构建主题间关联度查找表")
print("=" * 60)

# 预计算所有主题对之间的相关系数
topic_corr_table = np.zeros((K, K))
for i in range(K):
    name_i = f"Topic{i+1}"
    if name_i in topic_to_corr_idx:
        for j in range(K):
            name_j = f"Topic{j+1}"
            if name_j in topic_to_corr_idx:
                topic_corr_table[i, j] = corr_matrix[topic_to_corr_idx[name_i], topic_to_corr_idx[name_j]]

print(f"  主题间相关系数表 ({K}×{K}):")
for i in range(K):
    row_str = "  " + f"Topic{i+1}: " + " ".join(f"{topic_corr_table[i,j]:+.3f}" for j in range(K))
    print(row_str)

# ============================================================
# Step 4: 生成 concept_train_ctm_v1.json (9600 × 177)
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 生成 concept_train_ctm_v1.json (9600 × 177)")
print("=" * 60)

# 构建 sample_id → s_idx 映射（仅有效样本）
sample_id_to_sidx = {sid: i for i, sid in enumerate(sample_ids)}

# 全零样本的 concept 向量：均匀分布 1/N_ADJ
uniform_concept = [round(1.0 / N_ADJ, 6)] * N_ADJ

# 预计算每个形容词的主导主题对应的相关系数行
# adj_dominant[a] → 主题编号 → topic_corr_table 行
# 这样可以避免对每个 (sample, adj) 对都查表
adj_topic_corr_rows = topic_corr_table[adj_dominant, :]  # (N_ADJ, K)

# 遍历全部 9600 条样本
results = []
n_removed = 0
n_valid = 0

for orig_idx in range(len(train_data)):
    sid = f"sample_{orig_idx}"
    text = train_data[orig_idx]["content"]
    toxic = train_data[orig_idx]["toxic"]
    
    if sid in removed_sample_ids:
        # 全零样本：concept 设为均匀分布
        concept = uniform_concept[:]
        n_removed += 1
    elif sid in sample_id_to_sidx:
        # 有效样本：用主题间相关系数
        s_idx = sample_id_to_sidx[sid]
        s_topic = sample_dominant[s_idx]
        # 每个形容词与该样本的相关程度 = 主题间相关系数
        concept = [round(float(adj_topic_corr_rows[a_idx, s_topic]), 4) 
                   for a_idx in range(N_ADJ)]
        n_valid += 1
    else:
        # 理论上不应出现，但兜底处理
        concept = uniform_concept[:]
        n_removed += 1
    
    results.append({
        "content": text,
        "toxic": toxic,
        "concept": concept
    })

print(f"  生成样本数: {len(results)}")
print(f"  有效样本: {n_valid}")
print(f"  全零样本(均匀分布): {n_removed}")

# 保存
output_path = OUTPUT_DIR / "concept_train_ctm_v1.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n已保存: {output_path}")

# ============================================================
# Step 5: 统计信息
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 统计信息")
print("=" * 60)

# concept 值的分布
all_concepts = []
for r in results:
    all_concepts.extend(r["concept"])
all_concepts = np.array(all_concepts)

print(f"concept 值统计:")
print(f"  均值:   {all_concepts.mean():.4f}")
print(f"  中位数: {np.median(all_concepts):.4f}")
print(f"  最小值: {all_concepts.min():.4f}")
print(f"  最大值: {all_concepts.max():.4f}")
print(f"  标准差: {all_concepts.std():.4f}")

# 非零比例
non_zero = np.sum(all_concepts != 0)
print(f"  非零值比例: {100*non_zero/len(all_concepts):.2f}%")

# 唯一值数量
unique_vals = np.unique(all_concepts)
print(f"  唯一值数量: {len(unique_vals)}")

# 每个样本的 concept 中不同值的数量
concept_diversity = [len(set(r["concept"])) for r in results]
print(f"\n每个样本的 concept 唯一值数:")
print(f"  均值: {np.mean(concept_diversity):.2f}")
print(f"  最小: {np.min(concept_diversity)}")
print(f"  最大: {np.max(concept_diversity)}")

# 展示几个示例
print(f"\n概念向量示例 (前3个有效样本):")
shown = 0
for i, r in enumerate(results):
    if len(set(r["concept"])) > 1:  # 非均匀分布
        top_adj_indices = np.argsort(r["concept"])[::-1][:5]
        top_adjs = [(adj_name_list[j], r["concept"][j]) for j in top_adj_indices]
        print(f"  样本{i} (toxic={r['toxic']}): Top-5 形容词 = {top_adjs}")
        shown += 1
        if shown >= 3:
            break

print("\n完成！")
