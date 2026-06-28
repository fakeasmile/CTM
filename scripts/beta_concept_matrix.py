"""
beta_concept_matrix.py
-----------------------
方案B：利用 β 矩阵计算文本-形容词关联，绕过形容词 θ 坍缩问题。

核心思路：
  - 形容词在 β 矩阵中有自己的主题分布向量 β_adj (1×K)
  - 文本通过其词频对 β 矩阵加权求和得到主题向量 v_text = tf · β
  - 计算 v_text 与 β_adj 的余弦相似度

输出目录：output/experiments/

输出文件：
  - beta_adj_vectors.csv       : 形容词的 β 向量 (177×K)
  - beta_adj_similarity.csv    : 形容词间 β 向量余弦相似度 (177×177)
  - beta_concept_matrix.csv    : 文本-形容词 β 关联矩阵 (9058×177)
  - beta_adj_top_similar_pairs.csv : 形容词间 β 相似度 Top-20

用法：
  python scripts/beta_concept_matrix.py
"""

import csv
import os
import sys
import numpy as np
from pathlib import Path
from collections import OrderedDict

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path("d:/CTM")
MODEL_DIR = BASE_DIR / "output" / "ctm_model"
PREP_DIR = BASE_DIR / "output" / "preprocessing"
OUTPUT_DIR = BASE_DIR / "output" / "experiments"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Step 1: 读取数据
# ============================================================
print("=" * 60)
print("Step 1: 读取数据")
print("=" * 60)

# 1a. 读取 β 矩阵
print("读取 beta_matrix.csv ...")
beta_path = MODEL_DIR / "beta_matrix.csv"
beta_rows = []
beta_words = []

with open(beta_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    topic_names = header[1:]  # Topic1 ~ TopicK

    for row in reader:
        beta_words.append(row[0])
        beta_rows.append([float(x) for x in row[1:]])

beta_matrix = np.array(beta_rows)  # (V, K)
word2idx = {w: i for i, w in enumerate(beta_words)}
K = len(topic_names)
V = len(beta_words)
print(f"  β 矩阵: {V} × {K}")

# 1b. 读取形容词列表
print("读取形容词词典 ...")
adj_path = BASE_DIR / "data" / "raw" / "adjective" / "toxic_adjectives_v1.csv"
adj_list = []  # [(english, chinese, definition), ...]

with open(adj_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        adj_list.append((row["adjective"], row["chinese"], row["definition"]))

N_ADJ = len(adj_list)
print(f"  形容词数: {N_ADJ}")

# 1c. 读取元数据
print("读取元数据 ...")
meta_path = PREP_DIR / "dtm_metadata.csv"
sample_ids = []
sample_texts = []

with open(meta_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["doc_type"] == "sample":
            sample_ids.append(row["doc_id"])
            sample_texts.append(row["source"])

N_SAMPLE = len(sample_ids)
print(f"  样本数: {N_SAMPLE}")

# 1d. 读取 DTM triplet（稀疏格式）
print("读取 DTM triplet ...")
triplet_path = PREP_DIR / "dtm_triplet.csv"
triplet_doc_idx = []
triplet_term_idx = []
triplet_count = []

with open(triplet_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        triplet_doc_idx.append(int(row["doc_idx"]) - 1)  # R 1-indexed → Python 0-indexed
        triplet_term_idx.append(int(row["term_idx"]) - 1)
        triplet_count.append(int(row["count"]))

print(f"  Triplet 条目数: {len(triplet_doc_idx)}")

# 1e. 读取元数据获取样本的 doc_idx 映射
print("构建 doc_id → 行索引映射 ...")
doc_id_to_row = {}
with open(meta_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        doc_id_to_row[row["doc_id"]] = i

# 构建行索引 → 样本序号 的映射（只取 sample 类型）
row_to_sample_idx = {}
sample_counter = 0
with open(meta_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if row["doc_type"] == "sample":
            row_to_sample_idx[i] = sample_counter
            sample_counter += 1

# ============================================================
# Step 2: 提取形容词的 β 向量并验证区分度
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 提取形容词 β 向量 & 验证区分度")
print("=" * 60)

adj_chinese = [a[1] for a in adj_list]
adj_beta_vectors = np.zeros((N_ADJ, K))
adj_found = 0
adj_not_found = []

for i, (eng, chn, defn) in enumerate(adj_list):
    if chn in word2idx:
        adj_beta_vectors[i] = beta_matrix[word2idx[chn]]
        adj_found += 1
    else:
        adj_not_found.append(chn)
        # 尝试去掉"的"再查找
        chn_stripped = chn.rstrip("的")
        if chn_stripped in word2idx:
            adj_beta_vectors[i] = beta_matrix[word2idx[chn_stripped]]
            adj_found += 1
            adj_not_found.pop()  # 找到了，从 not_found 中移除

print(f"  在词表中找到的形容词: {adj_found}/{N_ADJ}")
if adj_not_found:
    print(f"  未找到的形容词 ({len(adj_not_found)}): {adj_not_found[:10]}...")

# 验证区分度：形容词间 β 向量的余弦相似度
def cosine_sim_matrix(A, B):
    """计算 A 和 B 之间的余弦相似度矩阵"""
    A_norm = A / (np.sqrt(np.sum(A**2, axis=1, keepdims=True)) + 1e-10)
    B_norm = B / (np.sqrt(np.sum(B**2, axis=1, keepdims=True)) + 1e-10)
    return A_norm @ B_norm.T

adj_beta_sim = cosine_sim_matrix(adj_beta_vectors, adj_beta_vectors)
upper_tri = adj_beta_sim[np.triu_indices_from(adj_beta_sim, k=1)]

print(f"\n形容词间 β 向量余弦相似度统计:")
print(f"  均值:   {upper_tri.mean():.4f}")
print(f"  中位数: {np.median(upper_tri):.4f}")
print(f"  最小值: {upper_tri.min():.4f}")
print(f"  最大值: {upper_tri.max():.4f}")
print(f"  标准差: {upper_tri.std():.4f}")

# 对比 K=10 时 θ 余弦相似度
print(f"\n  [对比] K=10 θ 余弦相似度统计:")
print(f"  均值: 0.950, 中位数: 0.997, 标准差: 0.140")

# 区间分布
sim_bins = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.01]
sim_labels = ['0~0.5', '0.5~0.8', '0.8~0.9', '0.9~0.95', '0.95~0.99', '0.99~1.0']
print(f"\n形容词间 β 相似度分布:")
for i in range(len(sim_labels)):
    count = np.sum((upper_tri >= sim_bins[i]) & (upper_tri < sim_bins[i+1]))
    pct = 100 * count / len(upper_tri)
    print(f"  {sim_labels[i]}: {count:>6d} ({pct:>5.1f}%)")

# 保存形容词 β 向量
beta_adj_path = OUTPUT_DIR / "beta_adj_vectors.csv"
with open(beta_adj_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["adj_id", "chinese", "english"] + topic_names)
    for i, (eng, chn, defn) in enumerate(adj_list):
        writer.writerow([f"adj_{i}", chn, eng] + [f"{x:.8e}" for x in adj_beta_vectors[i]])
print(f"\n已保存: {beta_adj_path}")

# 保存形容词间 β 相似度矩阵
adj_sim_path = OUTPUT_DIR / "beta_adj_similarity.csv"
with open(adj_sim_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([""] + [f"adj_{i}" for i in range(N_ADJ)])
    for i in range(N_ADJ):
        row = [f"adj_{i}"] + [f"{adj_beta_sim[i, j]:.6f}" for j in range(N_ADJ)]
        writer.writerow(row)
print(f"已保存: {adj_sim_path}")

# 保存 Top-20 最相似形容词对
sim_pairs = []
for i in range(N_ADJ - 1):
    for j in range(i + 1, N_ADJ):
        sim_pairs.append((i, j, adj_beta_sim[i, j]))
sim_pairs.sort(key=lambda x: x[2], reverse=True)

top_pairs_path = OUTPUT_DIR / "beta_adj_top_similar_pairs.csv"
with open(top_pairs_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["adj_i", "adj_j", "cosine_sim"])
    for idx_i, idx_j, sim_val in sim_pairs[:20]:
        writer.writerow([adj_chinese[idx_i], adj_chinese[idx_j], f"{sim_val:.6f}"])
print(f"已保存: {top_pairs_path}")

print(f"\nTop-5 最相似形容词对 (β 向量):")
for idx_i, idx_j, sim_val in sim_pairs[:5]:
    print(f"  {adj_chinese[idx_i]} ↔ {adj_chinese[idx_j]}: {sim_val:.4f}")

# ============================================================
# Step 3: 构建文本的 β 加权向量
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 构建文本 β 加权向量")
print("=" * 60)

# 从 triplet 构建样本的词频向量（稀疏）
# v_text = tf_text · β  → (1, V) × (V, K) = (1, K)
# 用稀疏方式实现，避免构建完整 DTM 矩阵

# 先收集每个样本的词频
from collections import defaultdict

sample_tf = defaultdict(lambda: defaultdict(int))  # sample_idx → {term_idx: count}

for t in range(len(triplet_doc_idx)):
    row_idx = triplet_doc_idx[t]
    term_idx = triplet_term_idx[t]
    count = triplet_count[t]
    if row_idx in row_to_sample_idx:
        s_idx = row_to_sample_idx[row_idx]
        sample_tf[s_idx][term_idx] = count

# 计算每个样本的 β 加权向量
text_vectors = np.zeros((N_SAMPLE, K))

for s_idx in range(N_SAMPLE):
    if s_idx in sample_tf and len(sample_tf[s_idx]) > 0:
        for term_idx, count in sample_tf[s_idx].items():
            text_vectors[s_idx] += count * beta_matrix[term_idx]

print(f"  文本 β 加权向量: {text_vectors.shape}")
print(f"  非零向量数: {np.sum(np.any(text_vectors > 0, axis=1))}")

# ============================================================
# Step 4: 计算文本-形容词 β 关联矩阵
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 计算文本-形容词 β 关联矩阵")
print("=" * 60)

concept_matrix = cosine_sim_matrix(text_vectors, adj_beta_vectors)

print(f"  概念矩阵维度: {concept_matrix.shape}")
print(f"  值范围: [{concept_matrix.min():.4f}, {concept_matrix.max():.4f}]")
print(f"  均值: {concept_matrix.mean():.4f}")
print(f"  中位数: {np.median(concept_matrix):.4f}")
print(f"  标准差: {concept_matrix.std():.4f}")
print(f"  >0.5 的比例: {100 * np.mean(concept_matrix > 0.5):.2f}%")
print(f"  >0.8 的比例: {100 * np.mean(concept_matrix > 0.8):.2f}%")

# 保存概念矩阵
cm_path = OUTPUT_DIR / "beta_concept_matrix.csv"
print(f"\n保存概念矩阵到 {cm_path} ...")

with open(cm_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_id", "text"] + [f"adj_{i}" for i in range(N_ADJ)])
    for s_idx in range(N_SAMPLE):
        row = [sample_ids[s_idx], sample_texts[s_idx]]
        row += [f"{concept_matrix[s_idx, a_idx]:.6f}" for a_idx in range(N_ADJ)]
        writer.writerow(row)

print(f"已保存: {cm_path}")

# ============================================================
# Step 5: 对比分析
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 对比分析 (β方案 vs θ方案)")
print("=" * 60)

# 读取原始 θ 概念矩阵
print("读取原始 θ 概念矩阵 ...")
theta_cm_path = MODEL_DIR / "concept_matrix_ctm.csv"
theta_cm_rows = []
with open(theta_cm_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        theta_cm_rows.append([float(x) for x in row[1:]])

theta_cm = np.array(theta_cm_rows)

print(f"\n{'指标':<25s} {'θ方案(余弦)':<15s} {'β方案(加权余弦)':<15s}")
print("-" * 55)
print(f"{'均值':<25s} {theta_cm.mean():<15.4f} {concept_matrix.mean():<15.4f}")
print(f"{'中位数':<25s} {np.median(theta_cm):<15.4f} {np.median(concept_matrix):<15.4f}")
print(f"{'标准差':<25s} {theta_cm.std():<15.4f} {concept_matrix.std():<15.4f}")
print(f"{'最小值':<25s} {theta_cm.min():<15.4f} {concept_matrix.min():<15.4f}")
print(f"{'最大值':<25s} {theta_cm.max():<15.4f} {concept_matrix.max():<15.4f}")
print(f"{'>0.5 比例':<25s} {100*np.mean(theta_cm>0.5):<15.2f}% {100*np.mean(concept_matrix>0.5):<14.2f}%")
print(f"{'>0.8 比例':<25s} {100*np.mean(theta_cm>0.8):<15.2f}% {100*np.mean(concept_matrix>0.8):<14.2f}%")

# 每个样本的 top-1 形容词变化
print(f"\n每个样本最高关联形容词:")
beta_top1 = concept_matrix.argmax(axis=1)
theta_top1 = theta_cm.argmax(axis=1)
changed = np.sum(beta_top1 != theta_top1)
print(f"  β方案和θ方案 top-1 不同的样本数: {changed}/{N_SAMPLE} ({100*changed/N_SAMPLE:.1f}%)")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
print(f"\n输出文件:")
print(f"  {OUTPUT_DIR / 'beta_adj_vectors.csv'}")
print(f"  {OUTPUT_DIR / 'beta_adj_similarity.csv'}")
print(f"  {OUTPUT_DIR / 'beta_adj_top_similar_pairs.csv'}")
print(f"  {OUTPUT_DIR / 'beta_concept_matrix.csv'}")
