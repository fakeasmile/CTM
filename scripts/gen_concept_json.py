"""
gen_concept_json.py
--------------------
生成 concept_train_ctm_v1.json

方法：
  1. 找到文本的概率最大主题 t_text
  2. 找到形容词的概率最大主题 t_adj
  3. 用 Σ 协方差矩阵中 t_text 和 t_adj 之间的相关系数
     作为该文本和该形容词之间的相关程度
  4. 若 t_text 或 t_adj 为 Topic10（不在 Σ 中），相关性设为 0

注意：Σ 是 (K-1)×(K-1) 矩阵 (logistic-normal 参数化)，
      Topic1~Topic9 之间的协方差/相关性可直接读取，
      Topic10 不在 Σ 中，无法计算其与其他主题的相关性。

全零样本处理：
  分词后因低频词过滤导致 DTM 全零的样本（542条），无法由 CTM 推断主题，
  其 concept 向量设为均匀分布 1/N_ADJ，表示无先验信息时对所有形容词关联相同。

输出：output/experiments/concept_train_ctm_v1.json
  [{"content": "...", "toxic": 0, "concept": [0.12, -0.03, ...]}, ...]
  维度：9600 × 177
"""

import json
import csv
import numpy as np
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path("d:/CTM")
MODEL_DIR = BASE_DIR / "output" / "ctm_model"
PREP_DIR = BASE_DIR / "output" / "preprocessing"
DATA_DIR = BASE_DIR / "data" / "raw" / "TOXICN"
OUTPUT_DIR = BASE_DIR / "output" / "experiments"

os.makedirs(OUTPUT_DIR, exist_ok=True) if not hasattr(Path, 'mkdir') else None
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import os

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
with open(PREP_DIR / "removed_docs.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["doc_type"] == "sample":
            removed_sample_ids.add(row["doc_id"])
print(f"  被移除的样本数: {len(removed_sample_ids)}")

# 构建原始样本索引映射: sample_N → train_data 中的索引
# sample_0 → train_data[0], sample_1 → train_data[1], ...
# 但有些 sample 被移除了，所以 sample_id 的编号不一定连续
# 需要从 doc_id 中提取原始索引
def sample_id_to_original_idx(sid):
    """sample_N → 原始 train.json 中的索引 N"""
    return int(sid.split("_")[1])

# 1c. 读取 theta_sample.csv
print("读取 theta_sample.csv ...")
sample_ids = []
theta_sample = []
with open(MODEL_DIR / "theta_sample.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    K = len(header) - 1  # 主题数
    topic_names = header[1:]
    for row in reader:
        sample_ids.append(row[0])
        theta_sample.append([float(x) for x in row[1:]])
theta_sample = np.array(theta_sample)  # (N_sample, K)
N_SAMPLE = len(sample_ids)
print(f"  θ_sample: {theta_sample.shape}, K={K}")

# 1d. 读取 theta_adj.csv
print("读取 theta_adj.csv ...")
adj_ids = []
adj_chinese = []
theta_adj = []
with open(MODEL_DIR / "theta_adj.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        adj_ids.append(row[0])
        theta_adj.append([float(x) for x in row[1:]])
theta_adj = np.array(theta_adj)  # (N_adj, K)
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
sigma = np.array(sigma_rows)  # (K-1, K-1)

# 计算相关系数矩阵
diag = np.sqrt(np.diag(sigma))
# 避免除以0
diag[diag == 0] = 1e-10
corr_matrix = sigma / np.outer(diag, diag)
np.fill_diagonal(corr_matrix, 1.0)

K_minus_1 = corr_matrix.shape[0]  # K-1
print(f"  Σ 维度: {sigma.shape[0]}×{sigma.shape[1]}")
print(f"  相关系数矩阵维度: {corr_matrix.shape[0]}×{corr_matrix.shape[1]}")
print(f"  Σ 中包含的主题: {sigma_topic_names}")

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

# 样本的主导主题（1-indexed: Topic1, Topic2, ..., TopicK）
sample_dominant = theta_sample.argmax(axis=1)  # 0-indexed
print(f"  样本主导主题分布:")
from collections import Counter
sample_topic_counter = Counter(sample_dominant)
for t in sorted(sample_topic_counter.keys()):
    print(f"    Topic{t+1}: {sample_topic_counter[t]} ({100*sample_topic_counter[t]/N_SAMPLE:.1f}%)")

# 形容词的主导主题
adj_dominant = theta_adj.argmax(axis=1)  # 0-indexed
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

# 关联度 = corr_matrix 中两个主题的相关系数
# 但 Σ 只有 Topic1~Topic9，没有 Topic10
# 当任一主题为 Topic10 时，无法从 Σ 获取相关性，设为 0

def get_topic_correlation(topic_i_0idx, topic_j_0idx):
    """
    获取两个主题之间的相关系数。
    topic_i_0idx, topic_j_0idx: 0-indexed 主题编号
    返回: 相关系数，若任一主题为 Topic10 (0-indexed=9) 则返回 0.0
    """
    # 0-indexed → Topic 名称
    name_i = f"Topic{topic_i_0idx + 1}"
    name_j = f"Topic{topic_j_0idx + 1}"
    
    # 如果任一主题不在 Σ 中
    if name_i not in topic_to_corr_idx or name_j not in topic_to_corr_idx:
        return 0.0
    
    idx_i = topic_to_corr_idx[name_i]
    idx_j = topic_to_corr_idx[name_j]
    return corr_matrix[idx_i, idx_j]

# 预计算所有主题对之间的相关系数
topic_corr_table = np.zeros((K, K))  # 10×10, Topic10 行列全为0
for i in range(K):
    for j in range(K):
        topic_corr_table[i, j] = get_topic_correlation(i, j)

print(f"  主题间相关系数表 ({K}×{K}):")
for i in range(K):
    row_str = "  " + f"Topic{i+1}: " + " ".join(f"{topic_corr_table[i,j]:+.3f}" for j in range(K))
    print(row_str)

# ============================================================
# Step 4: 生成 concept_train_ctm_v1.json
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 生成 concept_train_ctm_v1.json")
print("=" * 60)

# ---- 构建 sample_id → s_idx 映射（仅有效样本）----
sample_id_to_sidx = {sid: i for i, sid in enumerate(sample_ids)}

# ---- 全零样本的 concept 向量：均匀分布 1/N_ADJ ----
uniform_concept = [round(1.0 / N_ADJ, 6)] * N_ADJ

# ---- 遍历全部 9600 条样本 ----
results = []
n_removed = 0

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
        concept = []
        for a_idx in range(N_ADJ):
            a_topic = adj_dominant[a_idx]
            corr_val = topic_corr_table[s_topic, a_topic]
            concept.append(round(float(corr_val), 4))
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
print(f"  其中有效样本: {len(results) - n_removed}")
print(f"  其中全零样本(均匀分布): {n_removed}")

# 保存
output_path = OUTPUT_DIR / "concept_train_ctm_v1.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n已保存: {output_path}")

# ============================================================
# Step 5: 简要统计
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
print(f"  唯一值: {sorted(unique_vals)}")

# 每个样本的 concept 中不同值的数量
concept_diversity = [len(set(r["concept"])) for r in results]
print(f"\n每个样本的 concept 唯一值数:")
print(f"  均值: {np.mean(concept_diversity):.2f}")
print(f"  最小: {np.min(concept_diversity)}")
print(f"  最大: {np.max(concept_diversity)}")

print("\n完成！")
