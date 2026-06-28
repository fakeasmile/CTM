"""
topic_correlation.py
--------------------
从 CTM 的 Σ 协方差矩阵计算主题间相关性（Pearson 相关系数），
并输出到 output/experiments/ 目录。

CTM 的 Σ 是 logistic-normal 参数化下 (K-1)×(K-1) 的协方差矩阵，
其对角线元素为各主题的方差，非对角线元素为主题间的协方差。
相关系数 = cov(i,j) / sqrt(var(i) * var(j))

输出文件：
  - topic_correlation.csv  : 主题间相关系数矩阵 (K-1)×(K-1)
  - topic_top_correlations.csv : 主题对相关性排序（从最强正相关到最强负相关）

用法：
  python scripts/topic_correlation.py
"""

import csv
import os
import numpy as np
from pathlib import Path

BASE_DIR = Path("d:/CTM")
MODEL_DIR = BASE_DIR / "output" / "ctm_model"
OUTPUT_DIR = BASE_DIR / "output" / "experiments"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 1. 读取 sigma_matrix.csv ----
print("读取 sigma_matrix.csv ...")
sigma_path = MODEL_DIR / "sigma_matrix.csv"

with open(sigma_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    topic_names = header[1:]  # Topic1 ~ Topic(K-1)

    sigma_rows = []
    for row in reader:
        sigma_rows.append([float(x) for x in row[1:]])

sigma = np.array(sigma_rows)
K_minus_1 = sigma.shape[0]
print(f"  Σ 维度: {sigma.shape[0]} × {sigma.shape[1]} (K-1 = {K_minus_1})")

# ---- 2. 计算相关系数矩阵 ----
# corr(i,j) = cov(i,j) / sqrt(var(i) * var(j))
diag = np.sqrt(np.diag(sigma))
corr = sigma / np.outer(diag, diag)

# 确保对角线为 1（数值精度）
np.fill_diagonal(corr, 1.0)

print(f"  相关系数矩阵维度: {corr.shape}")

# ---- 3. 读取主题关键词（用于辅助解释）----
print("读取主题关键词 ...")
topic_terms_path = MODEL_DIR / "topic_top_terms.csv"
topic_keywords = {}

with open(topic_terms_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        topic_name = row["Topic"]
        # 取前5个关键词
        kw = [row[f"Rank{i}"] for i in range(1, 6)]
        topic_keywords[topic_name] = ", ".join(kw)

# ---- 4. 输出相关系数矩阵 ----
corr_path = OUTPUT_DIR / "topic_correlation.csv"

with open(corr_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([""] + topic_names)
    for i in range(K_minus_1):
        row = [topic_names[i]]
        for j in range(K_minus_1):
            row.append(f"{corr[i, j]:.6f}")
        writer.writerow(row)

print(f"已保存: {corr_path}")

# ---- 5. 输出主题对相关性排序 ----
pairs = []
for i in range(K_minus_1):
    for j in range(i + 1, K_minus_1):
        pairs.append({
            "topic_i": topic_names[i],
            "topic_j": topic_names[j],
            "keywords_i": topic_keywords.get(topic_names[i], ""),
            "keywords_j": topic_keywords.get(topic_names[j], ""),
            "correlation": corr[i, j]
        })

# 按相关性降序排列
pairs.sort(key=lambda x: x["correlation"], reverse=True)

top_corr_path = OUTPUT_DIR / "topic_top_correlations.csv"

with open(top_corr_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["topic_i", "topic_j", "correlation", "keywords_i", "keywords_j"])

    for p in pairs:
        writer.writerow([
            p["topic_i"], p["topic_j"],
            f"{p['correlation']:.6f}",
            p["keywords_i"], p["keywords_j"]
        ])

print(f"已保存: {top_corr_path}")

# ---- 6. 控制台输出关键信息 ----
print(f"\n{'='*60}")
print("主题间相关性分析 (K-1 = {})".format(K_minus_1))
print(f"{'='*60}")

# Top-5 正相关
print("\n最强正相关 Top-5:")
for p in pairs[:5]:
    print(f"  {p['topic_i']} ↔ {p['topic_j']}: {p['correlation']:.4f}")
    print(f"    {p['topic_i']}: {p['keywords_i']}")
    print(f"    {p['topic_j']}: {p['keywords_j']}")

# Top-5 负相关
print("\n最强负相关 Top-5:")
for p in pairs[-5:]:
    print(f"  {p['topic_i']} ↔ {p['topic_j']}: {p['correlation']:.4f}")
    print(f"    {p['topic_i']}: {p['keywords_i']}")
    print(f"    {p['topic_j']}: {p['keywords_j']}")

# 统计
corr_vals = [p["correlation"] for p in pairs]
print(f"\n相关系数统计:")
print(f"  均值: {np.mean(corr_vals):.4f}")
print(f"  正相关对数: {sum(1 for c in corr_vals if c > 0)}")
print(f"  负相关对数: {sum(1 for c in corr_vals if c < 0)}")
print(f"  |corr| > 0.3 的对数: {sum(1 for c in corr_vals if abs(c) > 0.3)}")
print(f"  |corr| > 0.5 的对数: {sum(1 for c in corr_vals if abs(c) > 0.5)}")

# 注意：Σ 是 (K-1)×(K-1)，缺少最后一个主题
print(f"\n注意: Σ 为 (K-1)×(K-1) 矩阵 (logistic-normal 参数化)，")
print(f"      不包含 Topic{K_minus_1 + 1} 的信息。")
print(f"      这是 CTM 模型的数学特性，最后一个主题由其余主题隐式决定。")
