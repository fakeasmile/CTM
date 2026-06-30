"""
train_ctm_python.py
-------------------
用 Python sklearn LDA 快速训练主题模型，
验证 enriched 形容词伪文档的区分度效果。

用法：
  python scripts/train_ctm_python.py
"""

import csv
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import datetime
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import LatentDirichletAllocation

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path("d:/CTM")
PREP_DIR = BASE_DIR / "output" / "preprocessing"
OUTPUT_DIR = BASE_DIR / "output" / "ctm_model"

K = 10
SEED = 42
TOP_N = 15
MAX_ITER = 50  # 减少迭代次数加快速度


def cosine_similarity(A, B):
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T


def entropy(p):
    p = np.array(p, dtype=np.float64)
    p = p / p.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Python LDA 训练（Enriched 形容词伪文档）")
    print("=" * 60)
    
    # ---- 1. 读取数据 ----
    print("\nStep 1: 读取数据")
    
    # 读取 triplet
    print("  读取 triplet...")
    doc_indices = []
    term_indices = []
    counts = []
    with open(PREP_DIR / "dtm_triplet.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            doc_indices.append(int(row[0]) - 1)  # 0-based
            term_indices.append(int(row[1]) - 1)
            counts.append(float(row[2]))
    
    # 读取词表
    print("  读取词表...")
    with open(PREP_DIR / "vocab.txt", "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
    
    # 读取元数据
    print("  读取元数据...")
    doc_ids = []
    doc_types = []
    doc_sources = []
    with open(PREP_DIR / "dtm_metadata.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_ids.append(row["doc_id"])
            doc_types.append(row["doc_type"])
            doc_sources.append(row["source"])
    
    n_docs = len(doc_ids)
    n_terms = len(vocab)
    print(f"  文档数: {n_docs}, 词数: {n_terms}")
    
    # 构建 sparse DTM
    print("  构建 sparse DTM...")
    dtm = csr_matrix((counts, (doc_indices, term_indices)), shape=(n_docs, n_terms))
    dtm.data = np.maximum(dtm.data, 0)  # 确保非负
    
    n_sample = sum(1 for d in doc_types if d == "sample")
    n_adj = sum(1 for d in doc_types if d == "adjective")
    print(f"  样本: {n_sample}, 形容词: {n_adj}")
    print(f"  非零元素: {dtm.nnz}, 稀疏度: {1 - dtm.nnz/(n_docs*n_terms):.4f}")
    
    # ---- 2. 训练 LDA ----
    print(f"\nStep 2: 训练 LDA (K={K}, max_iter={MAX_ITER})")
    start_time = datetime.now()
    
    lda = LatentDirichletAllocation(
        n_components=K,
        max_iter=MAX_ITER,
        learning_method='online',
        batch_size=256,
        random_state=SEED,
        verbose=1,
        n_jobs=-1
    )
    
    lda.fit(dtm)
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"\n训练完成！耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    
    # ---- 3. 提取矩阵 ----
    print("\nStep 3: 提取矩阵")
    
    # β 矩阵 (V × K)
    beta_raw = lda.components_.T  # (V, K)
    beta_matrix = beta_raw / beta_raw.sum(axis=0, keepdims=True)
    beta_df = pd.DataFrame(beta_matrix, index=vocab, columns=[f"Topic{i+1}" for i in range(K)])
    
    # θ 矩阵 (N × K)
    theta_all = lda.transform(dtm)
    theta_all = theta_all / theta_all.sum(axis=1, keepdims=True)
    
    sample_idx = [i for i, d in enumerate(doc_types) if d == "sample"]
    adj_idx = [i for i, d in enumerate(doc_types) if d == "adjective"]
    
    theta_sample = theta_all[sample_idx]
    theta_adj = theta_all[adj_idx]
    
    print(f"  β: {beta_matrix.shape}, θ_sample: {theta_sample.shape}, θ_adj: {theta_adj.shape}")
    
    # ---- 4. 保存矩阵 ----
    print("\nStep 4: 保存矩阵")
    
    beta_df.to_csv(OUTPUT_DIR / "beta_matrix.csv")
    
    pd.DataFrame(theta_sample, 
                 index=[doc_ids[i] for i in sample_idx],
                 columns=[f"Topic{i+1}" for i in range(K)]).to_csv(OUTPUT_DIR / "theta_sample.csv")
    
    pd.DataFrame(theta_adj,
                 index=[doc_ids[i] for i in adj_idx],
                 columns=[f"Topic{i+1}" for i in range(K)]).to_csv(OUTPUT_DIR / "theta_adj.csv")
    
    pd.DataFrame(theta_all,
                 index=doc_ids,
                 columns=[f"Topic{i+1}" for i in range(K)]).to_csv(OUTPUT_DIR / "theta_all.csv")
    
    # Σ 协方差
    sigma = np.cov(theta_all.T)
    pd.DataFrame(sigma,
                 index=[f"Topic{i+1}" for i in range(K)],
                 columns=[f"Topic{i+1}" for i in range(K)]).to_csv(OUTPUT_DIR / "sigma_matrix.csv")
    
    print("  已保存所有矩阵")
    
    # ---- 5. Top 词 ----
    print(f"\nStep 5: 各主题 Top-{TOP_N} 高频词")
    
    top_terms_data = []
    top_full_data = []
    
    for k in range(K):
        probs = lda.components_[k]
        probs_norm = probs / probs.sum()
        top_idx = np.argsort(probs_norm)[::-1]
        
        top_words = [vocab[i] for i in top_idx[:TOP_N]]
        print(f"  Topic {k+1}: {', '.join(top_words)}")
        
        top_terms_data.append({"Topic": f"Topic{k+1}", 
                               **{f"Rank{r+1}": vocab[top_idx[r]] for r in range(TOP_N)}})
        
        for rank, idx in enumerate(top_idx[:30]):
            top_full_data.append({"topic": k+1, "rank": rank+1, 
                                  "word": vocab[idx], "probability": probs_norm[idx]})
    
    pd.DataFrame(top_terms_data).to_csv(OUTPUT_DIR / "topic_top_terms.csv", index=False)
    pd.DataFrame(top_full_data).to_csv(OUTPUT_DIR / "topic_top_terms_full.csv", index=False)
    
    # ---- 6. 概念矩阵 ----
    print("\nStep 6: 计算文本-形容词余弦相似度")
    
    concept_matrix = cosine_similarity(theta_sample, theta_adj)
    print(f"  维度: {concept_matrix.shape}")
    print(f"  值范围: [{concept_matrix.min():.4f}, {concept_matrix.max():.4f}]")
    print(f"  均值: {concept_matrix.mean():.4f}, 中位数: {np.median(concept_matrix):.4f}")
    print(f"  >0.5: {100*np.mean(concept_matrix>0.5):.2f}%")
    print(f"  >0.8: {100*np.mean(concept_matrix>0.8):.2f}%")
    
    pd.DataFrame(concept_matrix,
                 index=[doc_ids[i] for i in sample_idx],
                 columns=[doc_ids[i] for i in adj_idx]).to_csv(OUTPUT_DIR / "concept_matrix_ctm.csv")
    
    # ---- 7. 形容词间相似度 ----
    print("\nStep 7: 形容词间主题分布相似度")
    
    adj_sim = cosine_similarity(theta_adj, theta_adj)
    adj_names = [doc_sources[i] for i in adj_idx]
    
    adj_sim_df = pd.DataFrame(adj_sim,
                               index=[doc_ids[i] for i in adj_idx],
                               columns=[doc_ids[i] for i in adj_idx])
    adj_sim_df.insert(0, "adj_name", adj_names)
    adj_sim_df.to_csv(OUTPUT_DIR / "adj_similarity.csv")
    
    # Top 最相似对
    sim_pairs = []
    for i in range(n_adj - 1):
        for j in range(i + 1, n_adj):
            sim_pairs.append((i, j, adj_sim[i, j]))
    sim_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("  最相似的形容词对 (Top-5):")
    for i, j, sim_val in sim_pairs[:5]:
        print(f"    {adj_names[i]} ↔ {adj_names[j]} : {sim_val:.4f}")
    
    top_pairs = [{"adj_i": adj_names[i], "adj_j": adj_names[j], "cosine_sim": sim_val}
                 for i, j, sim_val in sim_pairs[:20]]
    pd.DataFrame(top_pairs).to_csv(OUTPUT_DIR / "adj_top_similar_pairs.csv", index=False)
    
    # ---- 8. 形容词区分度分析 ----
    print("\n" + "=" * 60)
    print("形容词区分度分析")
    print("=" * 60)
    
    # 熵
    adj_entropies = [entropy(theta_adj[i]) for i in range(n_adj)]
    max_entropy = np.log2(K)
    print(f"\nθ 熵分析 (最大熵={max_entropy:.3f}):")
    print(f"  均值: {np.mean(adj_entropies):.3f}")
    print(f"  中位数: {np.median(adj_entropies):.3f}")
    print(f"  归一化熵均值: {np.mean(adj_entropies)/max_entropy:.4f}")
    
    # 主导主题
    dominant = np.argmax(theta_adj, axis=1)
    topic_counts = Counter(dominant)
    print(f"\n主导主题分布:")
    for t in sorted(topic_counts.keys()):
        pct = 100 * topic_counts[t] / n_adj
        print(f"  Topic{t+1}: {topic_counts[t]} ({pct:.1f}%)")
    
    max_topic_pct = 100 * max(topic_counts.values()) / n_adj
    print(f"\n坍缩指标 (最大主题占比): {max_topic_pct:.1f}%")
    
    # 形容词间相似度
    mask = ~np.eye(n_adj, dtype=bool)
    sim_values = adj_sim[mask]
    print(f"\n形容词间余弦相似度 (θ):")
    print(f"  均值: {np.mean(sim_values):.4f}")
    print(f"  中位数: {np.median(sim_values):.4f}")
    print(f"  >0.99: {100*np.mean(sim_values>0.99):.1f}%")
    
    # β分析
    adj_info = {}
    adj_csv_path = BASE_DIR / "data" / "raw" / "adjective" / "toxic_adjectives_v1.csv"
    with open(adj_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            adj_info[row["chinese"]] = row["adjective"]
    
    vocab_index = {w: i for i, w in enumerate(vocab)}
    adj_found = 0
    adj_norms = []
    js_values = []
    adj_beta_list = []
    adj_chinese_found = []
    
    for chinese, english in adj_info.items():
        if chinese in vocab_index:
            idx = vocab_index[chinese]
            beta_vec = lda.components_[:, idx]
            beta_vec = beta_vec / beta_vec.sum()
            adj_beta_list.append(beta_vec)
            adj_norms.append(np.linalg.norm(beta_vec))
            adj_chinese_found.append(chinese)
            adj_found += 1
    
    if adj_found > 0:
        adj_beta_arr = np.array(adj_beta_list)
        for i in range(len(adj_chinese_found)):
            for j in range(i+1, len(adj_chinese_found)):
                js_val = jensenshannon(adj_beta_arr[i], adj_beta_arr[j])
                js_values.append(js_val)
        
        js_values = np.array(js_values)
        print(f"\n形容词β向量 (在词表中: {adj_found}/{len(adj_info)}):")
        print(f"  L2范数均值: {np.mean(adj_norms):.6f}")
        print(f"  L2范数中位数: {np.median(adj_norms):.6f}")
        print(f"  JS散度均值: {np.mean(js_values):.6f}")
        print(f"  JS散度中位数: {np.median(js_values):.6f}")
    
    # ---- 保存配置 ----
    config_df = pd.DataFrame({
        "parameter": ["K", "SEED", "TOP_N", "method", "max_iter", "input_dir", "output_dir", "start_time", "elapsed_sec"],
        "value": [str(K), str(SEED), str(TOP_N), "LDA(sklearn)", str(MAX_ITER),
                  str(PREP_DIR), str(OUTPUT_DIR), start_time.strftime("%Y-%m-%d %H:%M:%S"), f"{elapsed:.1f}"]
    })
    config_df.to_csv(OUTPUT_DIR / "model_config.csv", index=False)
    
    # ---- 总结 ----
    print("\n" + "=" * 60)
    print("总结对比")
    print("=" * 60)
    print(f"""
指标                          原始版本(之前)    Enriched版本
─────────────────────────────────────────────────────
θ 坍缩率 (最大主题占比)       97.2%            {max_topic_pct:.1f}%
θ 余弦相似度中位数            0.997            {np.median(sim_values):.4f}
θ 归一化熵均值                极低             {np.mean(adj_entropies)/max_entropy:.4f}
概念矩阵中位数                0.0007           {np.median(concept_matrix):.4f}
""")
    
    if max_topic_pct < 80 and np.median(sim_values) < 0.99:
        print("✓ Enriched 版本显著改善了形容词区分度！")
    elif max_topic_pct < 90:
        print("△ 有一定改善，但仍有提升空间")
    else:
        print("✗ 区分度改善有限")
    
    print(f"\n所有输出已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
