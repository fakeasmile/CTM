"""
sample_topic_assignment.py
--------------------------
从 theta_sample.csv 中提取每个样本的主导主题及其概率，
并关联原始文本，输出到 output/experiments/ 目录。

输出文件：sample_topic_assignment.csv
  - sample_id   : 样本ID
  - text        : 样本原始文本
  - dominant_topic : 主导主题 (Topic1~TopicK)
  - topic_prob  : 属于该主题的概率值

用法：
  python scripts/sample_topic_assignment.py
"""

import csv
import os
from pathlib import Path

BASE_DIR = Path("d:/CTM")
MODEL_DIR = BASE_DIR / "output" / "ctm_model"
PREP_DIR = BASE_DIR / "output" / "preprocessing"
OUTPUT_DIR = BASE_DIR / "output" / "experiments"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 1. 读取 theta_sample.csv ----
print("读取 theta_sample.csv ...")
theta_path = MODEL_DIR / "theta_sample.csv"
samples = []

with open(theta_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    topic_names = header[1:]  # Topic1, Topic2, ..., TopicK

    for row in reader:
        sample_id = row[0]
        probs = [float(x) for x in row[1:]]
        max_idx = probs.index(max(probs))
        dominant_topic = topic_names[max_idx]
        topic_prob = probs[max_idx]
        samples.append({
            "sample_id": sample_id,
            "dominant_topic": dominant_topic,
            "topic_prob": topic_prob
        })

print(f"  读取 {len(samples)} 个样本")

# ---- 2. 读取元数据，获取原始文本 ----
print("读取元数据 (dtm_metadata.csv) ...")
meta_path = PREP_DIR / "dtm_metadata.csv"
text_map = {}

with open(meta_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["doc_type"] == "sample":
            text_map[row["doc_id"]] = row["source"]

print(f"  读取 {len(text_map)} 条样本文本")

# ---- 3. 合并并输出 ----
output_path = OUTPUT_DIR / "sample_topic_assignment.csv"

with open(output_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_id", "text", "dominant_topic", "topic_prob"])

    matched = 0
    for s in samples:
        sid = s["sample_id"]
        text = text_map.get(sid, "")
        if text:
            matched += 1
        writer.writerow([sid, text, s["dominant_topic"], f"{s['topic_prob']:.6f}"])

print(f"\n已保存: {output_path}")
print(f"  总样本数: {len(samples)}")
print(f"  匹配文本数: {matched}")

# ---- 4. 简要统计 ----
from collections import Counter
topic_counter = Counter(s["dominant_topic"] for s in samples)
print(f"\n各主题样本分布:")
for topic in sorted(topic_counter.keys()):
    count = topic_counter[topic]
    pct = 100 * count / len(samples)
    print(f"  {topic}: {count:>5d} ({pct:>5.1f}%)")
