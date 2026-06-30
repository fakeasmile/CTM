"""
prepare_dtm.py
--------------
将 TOXICN 训练集样本 + 形容词伪文档进行分词、构建 DocumentTermMatrix，
导出为 CSV 供 R 的 topicmodels 包读取。

输出目录：output/preprocessing/
  - dtm.csv              : DocumentTermMatrix (行=文档, 列=词, 值=词频)
  - dtm_triplet.csv      : DTM 稀疏格式 (doc_idx, term_idx, count)
  - dtm_metadata.csv     : 文档元数据 (doc_id, doc_type, source, token_count)
  - vocab.txt            : 词表 (每行一个词)
  - custom_dict.txt      : jieba 自定义词典
  - doc_freq.csv         : 词的文档频率统计
  - preprocessing.log    : 运行日志

用法：
  python scripts/prepare_dtm.py
"""

import json
import csv
import os
import re
import sys
import logging
from datetime import datetime
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ============================================================
# 配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "output" / "preprocessing"

TRAIN_JSON = DATA_DIR / "TOXICN" / "train.json"
# 优先使用 enriched 版本（含丰富伪文档），若不存在则回退到原始版本
ADJ_CSV_ENRICHED = DATA_DIR / "adjective" / "toxic_adjectives_v1_enriched.csv"
ADJ_CSV = ADJ_CSV_ENRICHED if ADJ_CSV_ENRICHED.exists() else DATA_DIR / "adjective" / "toxic_adjectives_v1.csv"
STOPWORDS_FILE = BASE_DIR / "stopwords" / "hit_stopwords.txt"

# 分词参数
MIN_WORD_LEN = 2        # 保留2字以上的词
MIN_DOC_FREQ = 10        # 词在至少10篇文档中出现才保留（减少词表加速CTM训练）

# ============================================================
# 日志设置：同时输出到控制台和文件
# ============================================================

def setup_logger(log_path):
    """配置日志，同时输出到控制台和文件"""
    logger = logging.getLogger("prepare_dtm")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除已有 handler

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件 handler
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

# ============================================================
# 停用词表：从文件加载
# ============================================================

def load_stopwords(filepath):
    """加载停用词表文件"""
    stopwords = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
    return stopwords

STOPWORDS = load_stopwords(STOPWORDS_FILE)

# ============================================================
# 分词函数（jieba）
# ============================================================

def init_jieba(adj_df, custom_dict_path, log):
    """初始化 jieba 分词器，将形容词加入自定义词典"""
    import jieba

    with open(custom_dict_path, "w", encoding="utf-8") as f:
        for _, row in adj_df.iterrows():
            chinese = row["chinese"].strip()
            if chinese:
                f.write(f"{chinese} 100 n\n")
                if len(chinese) > 2:
                    f.write(f"{chinese[:2]} 50 n\n")

    jieba.load_userdict(str(custom_dict_path))
    log.info(f"已加载自定义词典: {custom_dict_path}")
    return jieba


def tokenize(text, jieba_mod):
    """对文本进行分词，返回词列表"""
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


# ============================================================
# 主流程
# ============================================================

def load_adjectives(adj_csv_path, log):
    """读取形容词词典"""
    import pandas as pd
    df = pd.read_csv(adj_csv_path)
    log.info(f"读取形容词词典: {len(df)} 个形容词")
    return df


def load_toxicn(train_json_path, log):
    """读取 TOXICN 训练集"""
    with open(train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info(f"读取 TOXICN 训练集: {len(data)} 条样本")
    return data


def build_pseudo_docs(adj_df):
    """为每个形容词构建伪文档。
    如果存在 pseudo_doc 列（enriched版本），优先使用它作为伪文档内容；
    否则回退到 chinese + definition 的简单拼接。
    """
    has_pseudo_doc = "pseudo_doc" in adj_df.columns
    pseudo_docs = []
    for _, row in adj_df.iterrows():
        chinese = str(row["chinese"]).strip()
        definition = str(row["definition"]).strip()
        if has_pseudo_doc and not (isinstance(row.get("pseudo_doc"), float) and row.get("pseudo_doc") != row.get("pseudo_doc")):
            raw_text = str(row["pseudo_doc"]).strip()
        else:
            raw_text = f"{chinese} {definition}"
        pseudo_docs.append({
            "doc_id": f"adj_{_}",
            "doc_type": "adjective",
            "source": chinese,
            "raw_text": raw_text
        })
    return pseudo_docs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化日志
    log_path = OUTPUT_DIR / "preprocessing.log"
    log = setup_logger(log_path)

    log.info("=" * 60)
    log.info("prepare_dtm.py 开始运行")
    log.info("=" * 60)

    # 记录配置参数
    log.info(f"TRAIN_JSON    = {TRAIN_JSON}")
    log.info(f"ADJ_CSV       = {ADJ_CSV}")
    log.info(f"  (enriched版本: {ADJ_CSV_ENRICHED.exists()})")
    log.info(f"STOPWORDS_FILE= {STOPWORDS_FILE}")
    log.info(f"OUTPUT_DIR    = {OUTPUT_DIR}")
    log.info(f"MIN_WORD_LEN  = {MIN_WORD_LEN}")
    log.info(f"MIN_DOC_FREQ  = {MIN_DOC_FREQ}")

    # ---- 1. 读取数据 ----
    log.info("=" * 60)
    log.info("Step 1: 读取数据")
    log.info("=" * 60)

    log.info(f"停用词数量: {len(STOPWORDS)}")

    adj_df = load_adjectives(ADJ_CSV, log)
    if "pseudo_doc" in adj_df.columns:
        log.info(f"  检测到 enriched 版本 (含 pseudo_doc 列)")
        avg_len = adj_df["pseudo_doc"].str.len().mean()
        log.info(f"  伪文档平均长度: {avg_len:.0f} 字")
    else:
        log.info(f"  使用原始版本 (仅 chinese + definition)")
    toxicn_data = load_toxicn(TRAIN_JSON, log)

    # ---- 2. 构造文档列表 ----
    log.info("=" * 60)
    log.info("Step 2: 构造文档列表（样本 + 形容词伪文档）")
    log.info("=" * 60)

    documents = []

    for i, item in enumerate(toxicn_data):
        documents.append({
            "doc_id": f"sample_{i}",
            "doc_type": "sample",
            "source": item.get("content", ""),
            "raw_text": item.get("content", "")
        })

    pseudo_docs = build_pseudo_docs(adj_df)
    documents.extend(pseudo_docs)

    log.info(f"总文档数: {len(documents)} (样本: {len(toxicn_data)}, 形容词: {len(pseudo_docs)})")

    # ---- 3. 分词 ----
    log.info("=" * 60)
    log.info("Step 3: 分词 (jieba + 自定义词典)")
    log.info("=" * 60)

    custom_dict_path = OUTPUT_DIR / "custom_dict.txt"
    jieba_mod = init_jieba(adj_df, custom_dict_path, log)

    all_tokenized = []
    for doc in tqdm(documents, desc="分词中"):
        tokens = tokenize(doc["raw_text"], jieba_mod)
        all_tokenized.append(tokens)

    log.info(f"分词完成，平均词数: {sum(len(t) for t in all_tokenized) / len(all_tokenized):.1f}")

    # ---- 4. 构建词表（过滤低频词）----
    log.info("=" * 60)
    log.info("Step 4: 构建词表（过滤低频词）")
    log.info("=" * 60)

    doc_freq = Counter()
    for tokens in all_tokenized:
        unique_tokens = set(tokens)
        for t in unique_tokens:
            doc_freq[t] += 1

    # 保存完整的文档频率统计
    doc_freq_path = OUTPUT_DIR / "doc_freq.csv"
    with open(doc_freq_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "doc_freq", "retained"])
        for w in sorted(doc_freq.keys()):
            retained = doc_freq[w] >= MIN_DOC_FREQ
            writer.writerow([w, doc_freq[w], retained])
    log.info(f"文档频率统计已保存: {doc_freq_path}")

    # 形容词必须强制保留在词表中（即使文档频率 < MIN_DOC_FREQ）
    adj_chinese_list = adj_df["chinese"].tolist()
    adj_words_set = set(adj_chinese_list)

    # 过滤低频词，但形容词强制保留
    vocab = sorted([w for w, freq in doc_freq.items()
                    if freq >= MIN_DOC_FREQ or w in adj_words_set])
    vocab_set = set(vocab)
    vocab_index = {w: i for i, w in enumerate(vocab)}

    # 统计因低频被过滤但仍被强制保留的形容词数
    adj_forced = [a for a in adj_chinese_list if a in vocab_set and doc_freq.get(a, 0) < MIN_DOC_FREQ]

    log.info(f"原始词数: {len(doc_freq)}")
    log.info(f"过滤后词数 (文档频率 >= {MIN_DOC_FREQ}): {len(vocab)}")
    log.info(f"其中强制保留的形容词: {len(adj_forced)}")

    # 检查形容词覆盖情况
    adj_in_vocab = [a for a in adj_chinese_list if a in vocab_set]
    adj_not_in_vocab = [a for a in adj_chinese_list if a not in vocab_set]
    log.info(f"形容词在词表中: {len(adj_in_vocab)}/{len(adj_chinese_list)}")
    if adj_not_in_vocab:
        log.warning(f"以下形容词不在词表中（分词时未被切出）: {adj_not_in_vocab[:10]}...")
        for a in adj_not_in_vocab:
            vocab.append(a)
        vocab = sorted(vocab)
        vocab_set = set(vocab)
        vocab_index = {w: i for i, w in enumerate(vocab)}
        log.info(f"已强制将所有形容词加入词表，最终词数: {len(vocab)}")

    # ---- 5. 构建 DTM ----
    log.info("=" * 60)
    log.info("Step 5: 构建 DocumentTermMatrix")
    log.info("=" * 60)

    n_docs = len(documents)
    n_terms = len(vocab)

    # 用稀疏方式存储，只记录非零值
    dtm_rows = []
    for tokens in all_tokenized:
        row = Counter()
        for t in tokens:
            if t in vocab_index:
                row[vocab_index[t]] += 1
        dtm_rows.append(row)

    # 统计并过滤全零行（CTM 要求每行至少一个非零值）
    zero_row_indices = [i for i, r in enumerate(dtm_rows) if len(r) == 0]
    removed_docs_info = []
    if zero_row_indices:
        log.warning(f"发现 {len(zero_row_indices)} 个全零文档，将被移除（CTM 要求每行至少一个非零值）")
        zero_sample = sum(1 for i in zero_row_indices if documents[i]["doc_type"] == "sample")
        zero_adj = sum(1 for i in zero_row_indices if documents[i]["doc_type"] == "adjective")
        log.info(f"  被移除: sample={zero_sample}, adjective={zero_adj}")

        # 记录被移除的文档信息
        for i in zero_row_indices:
            removed_docs_info.append({
                "doc_id": documents[i]["doc_id"],
                "doc_type": documents[i]["doc_type"],
                "source": documents[i]["source"][:100]
            })

        # 保留非零行
        keep_indices = [i for i in range(len(dtm_rows)) if i not in set(zero_row_indices)]
        dtm_rows = [dtm_rows[i] for i in keep_indices]
        documents = [documents[i] for i in keep_indices]
        n_docs = len(documents)
        log.info(f"  过滤后文档数: {n_docs}")

    # 保存被移除文档信息
    if removed_docs_info:
        removed_path = OUTPUT_DIR / "removed_docs.csv"
        with open(removed_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["doc_id", "doc_type", "source"])
            writer.writeheader()
            writer.writerows(removed_docs_info)
        log.info(f"被移除文档信息已保存: {removed_path}")

    total_nonzero = sum(len(r) for r in dtm_rows)
    sparsity = 1 - total_nonzero / (n_docs * n_terms)
    log.info(f"DTM 维度: {n_docs} × {n_terms}")
    log.info(f"非零元素: {total_nonzero}, 稀疏度: {sparsity:.4f}")

    # ---- 6. 导出 DTM 为 CSV ----
    log.info("=" * 60)
    log.info("Step 6: 导出 DTM 为 CSV")
    log.info("=" * 60)

    dtm_csv_path = OUTPUT_DIR / "dtm.csv"
    with open(dtm_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + vocab)
        for i, row in enumerate(dtm_rows):
            row_data = [0] * n_terms
            for col_idx, count in row.items():
                row_data[col_idx] = count
            writer.writerow([documents[i]["doc_id"]] + row_data)

    log.info(f"DTM 已保存: {dtm_csv_path}")

    # ---- 7. 导出元数据 ----
    log.info("=" * 60)
    log.info("Step 7: 导出元数据")
    log.info("=" * 60)

    meta_csv_path = OUTPUT_DIR / "dtm_metadata.csv"
    with open(meta_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "doc_type", "source", "token_count"])
        for i, doc in enumerate(documents):
            writer.writerow([
                doc["doc_id"],
                doc["doc_type"],
                doc["source"][:100],
                len(dtm_rows[i])
            ])

    log.info(f"元数据已保存: {meta_csv_path}")

    # ---- 8. 导出词表 ----
    log.info("=" * 60)
    log.info("Step 8: 导出词表")
    log.info("=" * 60)

    vocab_path = OUTPUT_DIR / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for w in vocab:
            f.write(w + "\n")

    log.info(f"词表已保存: {vocab_path}")

    # ---- 9. 导出紧凑格式 (triplet) ----
    log.info("=" * 60)
    log.info("Step 9: 导出紧凑格式 (triplet)")
    log.info("=" * 60)

    triplet_path = OUTPUT_DIR / "dtm_triplet.csv"
    with open(triplet_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_idx", "term_idx", "count"])
        for i, row in enumerate(dtm_rows):
            for col_idx, count in row.items():
                writer.writerow([i + 1, col_idx + 1, count])  # R 的索引从1开始

    log.info(f"Triplet 格式已保存: {triplet_path}")

    # ---- 总结 ----
    log.info("=" * 60)
    log.info("完成！输出文件总结")
    log.info("=" * 60)
    log.info(f"  {dtm_csv_path}      : DTM 完整矩阵 ({n_docs} × {n_terms})")
    log.info(f"  {triplet_path}      : DTM 稀疏格式 (triplet)")
    log.info(f"  {meta_csv_path}     : 文档元数据")
    log.info(f"  {vocab_path}        : 词表 ({n_terms} 个词)")
    log.info(f"  {custom_dict_path}  : jieba 自定义词典")
    log.info(f"  {doc_freq_path}     : 词的文档频率统计")
    if removed_docs_info:
        log.info(f"  {OUTPUT_DIR / 'removed_docs.csv'} : 被移除的文档信息")
    log.info(f"  {log_path}          : 运行日志")
    log.info(f"\n[R 读取建议] 使用 triplet 格式 + Matrix::sparseMatrix() 更高效")

    log.info(f"\n运行完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
