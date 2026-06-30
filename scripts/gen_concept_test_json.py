"""
gen_concept_test_json.py
------------------------
为 TOXICN 测试集生成形容词概念向量（2411 × 177）

流程：
  1. 加载已训练好的 LDA 模型（从 theta_sample + beta_matrix + sigma 重建）
  2. 对测试集文本分词 → DTM → lda.transform 得到 θ_test
  3. 用"主导主题 + 主题相关系数"方法生成 concept 向量
  4. 输出 concept_test_ctm_v1.json

用法：
  python scripts/gen_concept_test_json.py
"""

import json
import csv
import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path("d:/CTM")
MODEL_DIR = BASE_DIR / "output" / "ctm_model"
PREP_DIR = BASE_DIR / "output" / "preprocessing"
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "output" / "experiments"

K = 10
SEED = 42
MAX_ITER = 50

# ============================================================
# 加载停用词
# ============================================================
STOPWORDS_FILE = BASE_DIR / "stopwords" / "hit_stopwords.txt"
MIN_WORD_LEN = 2

def load_stopwords(filepath):
    stopwords = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
    return stopwords

STOPWORDS = load_stopwords(STOPWORDS_FILE)

def tokenize(text, jieba_mod):
    words = jieba_mod.lcut(text)
    filtered = []
    for w in words:
        w = w.strip()
        if len(w) < MIN_WORD_LEN:
            continue
        if w in STOPWORDS:
            continue
        if re.match(r"^[\d\s\W]+$", w) and not re.search(r"[\u4e00-\u9fff]", w):
            continue
        if re.match(r"^[a-zA-Z]$", w):
            continue
        filtered.append(w)
    return filtered


def cosine_similarity(A, B):
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("为测试集生成形容词概念向量 (2411 × 177)")
    print("=" * 60)

    # ============================================================
    # Step 1: 读取测试集数据
    # ============================================================
    print("\nStep 1: 读取测试集数据")

    test_json_path = DATA_DIR / "TOXICN" / "test.json"
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"  测试样本数: {len(test_data)}")

    # ============================================================
    # Step 2: 读取训练时的词表和模型参数
    # ============================================================
    print("\nStep 2: 读取词表和模型参数")

    # 读取词表
    vocab_path = PREP_DIR / "vocab.txt"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
    vocab_set = set(vocab)
    vocab_index = {w: i for i, w in enumerate(vocab)}
    n_terms = len(vocab)
    print(f"  词表大小: {n_terms}")

    # 读取 β 矩阵（词×主题），用于重建 LDA
    print("  读取 beta_matrix.csv ...")
    beta_matrix = []
    with open(MODEL_DIR / "beta_matrix.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        topic_names = header[1:]
        K = len(topic_names)
        for row in reader:
            beta_matrix.append([float(x) for x in row[1:]])
    beta_matrix = np.array(beta_matrix)  # (V, K)
    print(f"  β 矩阵: {beta_matrix.shape}")

    # 读取 θ_adj
    print("  读取 theta_adj.csv ...")
    theta_adj = []
    adj_ids = []
    with open(MODEL_DIR / "theta_adj.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            adj_ids.append(row[0])
            theta_adj.append([float(x) for x in row[1:]])
    theta_adj = np.array(theta_adj)
    N_ADJ = len(adj_ids)
    print(f"  θ_adj: {theta_adj.shape}")

    # 读取 Σ 协方差矩阵
    print("  读取 sigma_matrix.csv ...")
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

    # 计算主题间相关系数矩阵
    diag = np.sqrt(np.abs(np.diag(sigma)))
    diag[diag == 0] = 1e-10
    corr_matrix = sigma / np.outer(diag, diag)
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    topic_to_corr_idx = {name: i for i, name in enumerate(sigma_topic_names)}

    # 形容词词典
    adj_path = DATA_DIR / "adjective" / "toxic_adjectives_v1.csv"
    adj_name_list = []
    with open(adj_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            adj_name_list.append(row["chinese"])
    print(f"  形容词数: {len(adj_name_list)}")

    # ============================================================
    # Step 3: 对测试集分词 + 构建 DTM
    # ============================================================
    print("\nStep 3: 对测试集分词 + 构建 DTM")

    # 初始化 jieba（加载训练时的自定义词典）
    import jieba
    custom_dict_path = PREP_DIR / "custom_dict.txt"
    if custom_dict_path.exists():
        jieba.load_userdict(str(custom_dict_path))
        print(f"  已加载自定义词典: {custom_dict_path}")

    # 分词
    test_tokenized = []
    for item in test_data:
        tokens = tokenize(item["content"], jieba)
        test_tokenized.append(tokens)

    avg_tokens = np.mean([len(t) for t in test_tokenized])
    print(f"  分词完成，平均词数: {avg_tokens:.1f}")

    # 构建 DTM（使用训练时的词表）
    doc_indices = []
    term_indices = []
    counts = []
    zero_doc_indices = []  # 全零文档的索引

    for i, tokens in enumerate(test_tokenized):
        word_count = Counter()
        for t in tokens:
            if t in vocab_index:
                word_count[vocab_index[t]] += 1
        if len(word_count) == 0:
            zero_doc_indices.append(i)
            continue
        for term_idx, count in word_count.items():
            doc_indices.append(len(doc_indices))  # 重新编号
            term_indices.append(term_idx)
            counts.append(count)

    n_test = len(test_data)
    if len(doc_indices) > 0:
        # 需要重新编号 doc_indices，因为跳过了全零文档
        # 重新构建：非零文档按顺序编号
        doc_indices_new = []
        term_indices_new = []
        counts_new = []
        doc_counter = 0
        non_zero_mask = []

        for i, tokens in enumerate(test_tokenized):
            word_count = Counter()
            for t in tokens:
                if t in vocab_index:
                    word_count[vocab_index[t]] += 1
            if len(word_count) == 0:
                non_zero_mask.append(False)
                continue
            non_zero_mask.append(True)
            for term_idx, count in word_count.items():
                doc_indices_new.append(doc_counter)
                term_indices_new.append(term_idx)
                counts_new.append(count)
            doc_counter += 1

        dtm_test = csr_matrix((counts_new, (doc_indices_new, term_indices_new)),
                               shape=(doc_counter, n_terms))
    else:
        dtm_test = csr_matrix((0, n_terms))
        non_zero_mask = [False] * n_test

    non_zero_mask = np.array(non_zero_mask)
    print(f"  测试集 DTM: {dtm_test.shape}, 非零文档: {dtm_test.shape[0]}, 全零文档: {sum(~non_zero_mask)}")

    # ============================================================
    # Step 4: 用已训练的 LDA 模型推断测试集 θ
    # ============================================================
    print("\nStep 4: 用 LDA 推断测试集 θ")

    # 重建 LDA 模型（从 beta 矩阵恢复 components_）
    # sklearn LDA 的 components_ 是 (K, V) 的 β_raw（未归一化）
    # 我们可以从归一化的 beta_matrix 反推
    # 但更简单的方法是：直接重新训练一个 LDA，用训练集的 DTM
    # 然后用 transform 得到测试集的 θ
    
    # 最简单的方式：用 transform 直接推断
    # 但需要已拟合的 LDA 模型对象
    # 由于我们没有保存模型对象，需要重建

    # 方案：从 beta_matrix 重建 LDA 的 components_
    # beta_matrix[i,k] = P(word_i | topic_k)
    # LDA 的 components_ 存储的是未归一化的词-主题计数
    # 我们可以用 beta_matrix * 一个大常数来近似
    # 但 transform 方法会内部做归一化，所以只需要比例关系正确
    
    # 更可靠：用训练集 DTM 重新训练
    print("  读取训练集 DTM ...")
    train_triplet_path = PREP_DIR / "dtm_triplet.csv"
    train_doc_indices = []
    train_term_indices = []
    train_counts = []
    with open(train_triplet_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            train_doc_indices.append(int(row[0]) - 1)
            train_term_indices.append(int(row[1]) - 1)
            train_counts.append(float(row[2]))

    # 读取训练集元数据获取文档数
    train_meta_path = PREP_DIR / "dtm_metadata.csv"
    train_n_docs = 0
    with open(train_meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_n_docs += 1

    dtm_train = csr_matrix((train_counts, (train_doc_indices, train_term_indices)),
                            shape=(train_n_docs, n_terms))
    dtm_train.data = np.maximum(dtm_train.data, 0)

    print(f"  训练集 DTM: {dtm_train.shape}")

    # 训练 LDA
    print(f"  训练 LDA (K={K}, max_iter={MAX_ITER}) ...")
    lda = LatentDirichletAllocation(
        n_components=K,
        max_iter=MAX_ITER,
        learning_method='online',
        batch_size=256,
        random_state=SEED,
        n_jobs=-1
    )
    lda.fit(dtm_train)
    print("  LDA 训练完成")

    # 推断测试集 θ
    if dtm_test.shape[0] > 0:
        theta_test_nonzero = lda.transform(dtm_test)
        theta_test_nonzero = theta_test_nonzero / theta_test_nonzero.sum(axis=1, keepdims=True)
    else:
        theta_test_nonzero = np.array([])

    # 构建完整的 θ_test（全零文档用均匀分布）
    theta_test = np.ones((n_test, K)) / K  # 全零文档默认均匀分布
    nonzero_idx = 0
    for i in range(n_test):
        if non_zero_mask[i]:
            theta_test[i] = theta_test_nonzero[nonzero_idx]
            nonzero_idx += 1

    print(f"  θ_test: {theta_test.shape}")

    # ============================================================
    # Step 5: 用"主导主题 + 主题相关系数"生成 concept 向量
    # ============================================================
    print("\nStep 5: 生成概念向量 (2411 × 177)")

    # 形容词的主导主题
    adj_dominant = theta_adj.argmax(axis=1)

    # 主题间相关系数查找表
    topic_corr_table = np.zeros((K, K))
    for i in range(K):
        name_i = f"Topic{i+1}"
        if name_i in topic_to_corr_idx:
            for j in range(K):
                name_j = f"Topic{j+1}"
                if name_j in topic_to_corr_idx:
                    topic_corr_table[i, j] = corr_matrix[topic_to_corr_idx[name_i], topic_to_corr_idx[name_j]]

    # 预计算
    adj_topic_corr_rows = topic_corr_table[adj_dominant, :]  # (N_ADJ, K)

    # 测试集的主导主题
    test_dominant = theta_test.argmax(axis=1)
    print(f"  测试集主导主题分布:")
    test_topic_counter = Counter(test_dominant)
    for t in sorted(test_topic_counter.keys()):
        print(f"    Topic{t+1}: {test_topic_counter[t]} ({100*test_topic_counter[t]/n_test:.1f}%)")

    # 生成 concept 向量
    uniform_concept = [round(1.0 / N_ADJ, 6)] * N_ADJ

    results = []
    for i in range(n_test):
        text = test_data[i]["content"]
        toxic = test_data[i]["toxic"]

        if not non_zero_mask[i]:
            # 全零文档
            concept = uniform_concept[:]
        else:
            # 有效文档
            s_topic = test_dominant[i]
            concept = [round(float(adj_topic_corr_rows[a_idx, s_topic]), 4)
                       for a_idx in range(N_ADJ)]

        results.append({
            "content": text,
            "toxic": toxic,
            "concept": concept
        })

    print(f"  生成样本数: {len(results)}")
    print(f"  有效样本: {sum(non_zero_mask)}")
    print(f"  全零样本: {sum(~non_zero_mask)}")

    # 保存
    output_path = OUTPUT_DIR / "concept_test_ctm_v1.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n已保存: {output_path}")

    # ============================================================
    # Step 6: 统计
    # ============================================================
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)

    all_concepts = []
    for r in results:
        all_concepts.extend(r["concept"])
    all_concepts = np.array(all_concepts)

    print(f"concept 值统计:")
    print(f"  均值:   {all_concepts.mean():.4f}")
    print(f"  中位数: {np.median(all_concepts):.4f}")
    print(f"  最小值: {all_concepts.min():.4f}")
    print(f"  最大值: {all_concepts.max():.4f}")

    # 示例
    print(f"\n概念向量示例 (前3个有效样本):")
    shown = 0
    for i, r in enumerate(results):
        if len(set(r["concept"])) > 1:
            top_adj_indices = np.argsort(r["concept"])[::-1][:5]
            top_adjs = [(adj_name_list[j], r["concept"][j]) for j in top_adj_indices]
            print(f"  样本{i} (toxic={r['toxic']}): Top-5 形容词 = {top_adjs}")
            shown += 1
            if shown >= 3:
                break

    print("\n完成！")


if __name__ == "__main__":
    main()
