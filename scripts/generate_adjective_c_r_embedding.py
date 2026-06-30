"""生成形容词概念向量（LLM Embedding 语义空间映射）

核心思路：用 LLM 编码模式提取语义向量，替代 CTM 的主题分布，
在高维语义空间中度量文本与形容词之间的相关程度。

【与 CTM 方案的对应关系】
  CTM:  文本 → θ_text (K=10维) → 主题空间 ← θ_adj (K=10维) ← 形容词
                                      ↓
                              corr(θ_text, θ_adj) → 概念向量

  LLM:  文本 → h_text (D=4096维) → 语义空间 ← h_adj (D=4096维) ← 形容词
                                        ↓
                              cos(h_text, h_adj) → 概念向量

【为什么比直接评估好】
  - 直接评估：LLM 生成评分，1-5 离散值，区分度低，易坍缩
  - Embedding 映射：连续高维空间，区分度高，天然抗坍缩
  - 完美继承 CTM 架构思想：通过中间表示空间间接度量

【可复现性保证】
  - 编码模式不涉及采样，纯前向传播矩阵运算
  - 设置确定性模式：torch.use_deterministic_mode + CUBLAS_WORKSPACE_CONFIG
  - 保存中间结果（embedding 矩阵）到文件，下游只读文件

【流程】
  1. 加载 LLM + tokenizer
  2. 编码所有形容词（含定义）→ h_adj [V, D]
  3. 编码所有文本 → h_text [N, D]
  4. 计算 cosine similarity → concept_matrix [N, V]
  5. 保存结果

使用示例：
python scripts/generate_adjective_c_r_embedding.py --mode train --model_name glm-4-9b-chat
python scripts/generate_adjective_c_r_embedding.py --mode test --model_name Qwen2.5-7B-Instruct
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# =============================================================================
# 项目路径
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ADJ_DIR = DATA_DIR / "adjective"
TOXICN_DIR = DATA_DIR / "TOXICN"
OUTPUT_DIR = PROJECT_ROOT / "output" / "experiments"
MODELS_PATH = PROJECT_ROOT / "models"

# 设置确定性环境变量（在 import torch 之前）
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# =============================================================================
# 命令行参数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="生成形容词概念向量（LLM Embedding 语义空间映射）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='train: 生成训练集概念向量, test: 生成测试集概念向量')
    parser.add_argument('--model_name', type=str, required=True,
                        help='LLM模型名称（也是models/下的子目录名）')
    parser.add_argument('--adjective_name', type=str, default='toxic_adjectives_v1.csv',
                        help='形容词词典文件名，默认toxic_adjectives_v1.csv')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='文本编码的batch size，默认64')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU显存占用比例，默认0.9')
    parser.add_argument('--no_deterministic', action='store_true',
                        help='禁用确定性模式（可能轻微加速）')
    parser.add_argument('--save_embedding', action='store_true',
                        help='同时保存 embedding 矩阵（用于调试和复现）')
    return parser.parse_args()


# =============================================================================
# 模型加载配置表
# =============================================================================
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {
        "quantization": None,
        "is_multimodal": False,
        "max_length": 512,
    },
    "Qwen3.5-9B": {
        "quantization": "fp8",
        "is_multimodal": True,
        "max_length": 512,
    },
    "glm-4-9b-chat": {
        "quantization": None,
        "is_multimodal": False,
        "max_length": 512,
    },
    "deepseek-llm-7b-chat": {
        "quantization": None,
        "is_multimodal": False,
        "max_length": 512,
    },
    "Baichuan2-7B-Chat": {
        "quantization": None,
        "is_multimodal": False,
        "max_length": 512,
    },
    "Qwen3-8B": {
        "quantization": None,
        "is_multimodal": False,
        "max_length": 512,
    },
}


def get_model_loading_config(model_name: str) -> dict:
    """从 MODEL_LOADING_CONFIG 中获取模型加载配置，未知模型直接报错。"""
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(
            f"不支持的模型: {model_name}。请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


# =============================================================================
# 模型加载（编码模式）
# =============================================================================
def load_encoder_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.9):
    """加载 LLM 编码器模型和 tokenizer。

    使用 AutoModel（而非 AutoModelForCausalLM）以获取 hidden states。
    对于因果语言模型，AutoModel 会自动退回到 AutoModelForCausalLM 的
    基座部分（不含 lm_head），确保能提取中间层表示。

    Returns: (tokenizer, model, config)
    """
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = get_model_loading_config(model_name)

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    print(f"Loading model from {llm_path}")
    quantization = model_config["quantization"]

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32,  # 使用 float32 保证数值精度和可复现性
        "output_hidden_states": True,
    }

    if quantization is not None:
        from transformers import BitsAndBytesConfig
        if quantization == "fp8":
            # FP8 量化配置
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantization == "int4":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        print(f"  量化方式: {quantization}")
    else:
        print(f"  量化方式: 无量化 (float32)")

    model = AutoModel.from_pretrained(llm_path, **model_kwargs)

    if quantization is None:
        model = model.cuda()

    model.eval()
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  设备: {next(model.parameters()).device}")

    return tokenizer, model, model_config


# =============================================================================
# Embedding 提取
# =============================================================================
@torch.no_grad()
def encode_texts(texts: list[str], tokenizer, model, batch_size: int = 64,
                 max_length: int = 512, desc: str = "Encoding") -> np.ndarray:
    """批量编码文本，提取最后一层 hidden state 的 mean pooling 作为语义向量。

    Args:
        texts: 文本列表
        tokenizer: 分词器
        model: 编码器模型
        batch_size: 批量大小
        max_length: 最大序列长度
        desc: 进度条描述

    Returns:
        np.ndarray: [N, D] 的 embedding 矩阵
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i:i + batch_size]

        # 分词
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 前向传播
        outputs = model(**inputs)

        # Mean Pooling：对所有 token 的 hidden state 取平均（忽略 padding）
        # hidden_states: [batch, seq_len, hidden_dim]
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [batch, seq_len, 1]

        # 加权平均（只计算非 padding 位置）
        masked_hidden = last_hidden * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)  # [batch, hidden_dim]
        count = attention_mask.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
        embeddings = sum_hidden / count  # [batch, hidden_dim]

        all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def build_adj_texts(adj_df: pd.DataFrame) -> list[str]:
    """构建形容词的编码输入文本。

    将形容词的中文名和定义拼接，提供更丰富的语义信息：
        "{adj}：{definition}"

    这与 CTM 方案中用伪文档扩展形容词的思路一脉相承，
    但这里利用 LLM 自身来理解定义，无需人工模板。
    """
    texts = []
    for _, row in adj_df.iterrows():
        adj_cn = row["chinese"]
        definition = row.get("definition", "")
        if isinstance(definition, str) and definition.strip():
            texts.append(f"{adj_cn}：{definition}")
        else:
            texts.append(adj_cn)
    return texts


def compute_concept_matrix(h_text: np.ndarray, h_adj: np.ndarray) -> np.ndarray:
    """计算概念向量矩阵。

    concept[i, j] = cosine_similarity(h_text[i], h_adj[j])

    Args:
        h_text: [N, D] 文本 embedding 矩阵
        h_adj: [V, D] 形容词 embedding 矩阵

    Returns:
        np.ndarray: [N, V] 概念向量矩阵，值域 [0, 1]
    """
    # L2 归一化
    h_text_norm = h_text / np.linalg.norm(h_text, axis=1, keepdims=True).clip(min=1e-9)
    h_adj_norm = h_adj / np.linalg.norm(h_adj, axis=1, keepdims=True).clip(min=1e-9)

    # 矩阵乘法计算所有 cosine similarity
    concept_matrix = h_text_norm @ h_adj_norm.T  # [N, V]

    # 裁剪到 [-1, 1]（数值精度可能超出）
    concept_matrix = np.clip(concept_matrix, -1.0, 1.0)

    # 线性映射到 [0, 1]：将 [-1, 1] → [0, 1]
    # 这样 cos=1 → score=1.0 (高度相关), cos=-1 → score=0.0 (完全相反)
    concept_matrix = (concept_matrix + 1.0) / 2.0

    return concept_matrix


# =============================================================================
# 可复现性配置
# =============================================================================
def set_deterministic_mode():
    """设置 PyTorch 确定性模式，保证完全可复现。

    注意：确定性模式会略微降低性能（约 5-10%），
    但对于实验的可复现性至关重要。
    """
    try:
        torch.use_deterministic_mode(True)
        print("已启用 PyTorch 确定性模式")
    except Exception as e:
        print(f"警告：无法启用确定性模式: {e}")
        print("  GPU 浮点运算可能存在微小差异（<1e-7），不影响结果质量")


# =============================================================================
# 诊断：复现性校验
# =============================================================================
def verify_reproducibility(texts: list[str], tokenizer, model, n_check: int = 5):
    """验证编码结果的可复现性：对前 n_check 条文本编码两次，比较差异。

    如果差异 < 1e-5，说明可复现性良好。
    """
    check_texts = texts[:n_check]
    emb1 = encode_texts(check_texts, tokenizer, model, batch_size=n_check, desc="Verify #1")
    emb2 = encode_texts(check_texts, tokenizer, model, batch_size=n_check, desc="Verify #2")

    max_diff = np.abs(emb1 - emb2).max()
    mean_diff = np.abs(emb1 - emb2).mean()

    print(f"\n{'=' * 60}")
    print("可复现性校验")
    print(f"{'=' * 60}")
    print(f"  校验样本数: {n_check}")
    print(f"  最大差异: {max_diff:.2e}")
    print(f"  平均差异: {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("  ✓ 可复现性优秀（差异 < 1e-5）")
    elif max_diff < 1e-3:
        print("  ⚠ 可复现性可接受（差异 < 1e-3），存在微小浮点差异")
    else:
        print("  ✗ 可复现性较差，请检查确定性模式是否正确启用")
    print(f"{'=' * 60}")

    return max_diff


# =============================================================================
# 诊断：概念向量质量评估
# =============================================================================
def evaluate_concept_quality(concept_matrix: np.ndarray, adj_names: list[str],
                             data: list[dict], label: str = ""):
    """评估概念向量的质量：区分度、稀疏性、语义合理性。"""
    N, V = concept_matrix.shape

    print(f"\n{'=' * 60}")
    print(f"概念向量质量评估 {label}")
    print(f"{'=' * 60}")
    print(f"矩阵形状: [{N}, {V}]")

    # 1. 基础统计
    print(f"\n基础统计:")
    print(f"  均值:   {concept_matrix.mean():.4f}")
    print(f"  中位数: {np.median(concept_matrix):.4f}")
    print(f"  标准差: {concept_matrix.std():.4f}")
    print(f"  最小值: {concept_matrix.min():.4f}")
    print(f"  最大值: {concept_matrix.max():.4f}")

    # 2. 区分度指标
    # 每条文本的概念向量的标准差（衡量该文本对不同形容词的区分程度）
    per_sample_std = concept_matrix.std(axis=1)
    print(f"\n区分度指标:")
    print(f"  逐样本标准差 均值: {per_sample_std.mean():.4f}")
    print(f"  逐样本标准差 最小: {per_sample_std.min():.4f}")
    print(f"  逐样本标准差 最大: {per_sample_std.max():.4f}")

    # 每条文本中 Top-1 和 Bottom-1 的差值
    per_sample_range = concept_matrix.max(axis=1) - concept_matrix.min(axis=1)
    print(f"  逐样本极差 均值: {per_sample_range.mean():.4f}")
    print(f"  逐样本极差 最小: {per_sample_range.min():.4f}")

    # 3. 稀疏性
    # 高分比例（score > 0.6，对应原始 cosine > 0.2）
    high_score_ratio = (concept_matrix > 0.6).mean()
    low_score_ratio = (concept_matrix < 0.4).mean()
    print(f"\n稀疏性:")
    print(f"  高分区(>0.6)比例: {high_score_ratio:.2%}")
    print(f"  低分区(<0.4)比例: {low_score_ratio:.2%}")

    # 4. 与毒性标签的一致性（如果数据中有 toxic 字段）
    if data and "toxic" in data[0]:
        toxic_mask = np.array([d["toxic"] == 1 for d in data[:N]])
        non_toxic_mask = ~toxic_mask

        if toxic_mask.sum() > 0 and non_toxic_mask.sum() > 0:
            # 找出"攻击性"相关的形容词索引
            attack_adjs = [i for i, name in enumerate(adj_names)
                          if any(kw in name for kw in ["攻击", "辱骂", "威胁", "暴力", "欺凌", "歧视"])]
            positive_adjs = [i for i, name in enumerate(adj_names)
                            if any(kw in name for kw in ["包容", "赞赏", "体贴", "感激", "调和"])]

            if attack_adjs:
                toxic_attack = concept_matrix[toxic_mask][:, attack_adjs].mean()
                nontoxic_attack = concept_matrix[non_toxic_mask][:, attack_adjs].mean()
                print(f"\n语义一致性（攻击性形容词）:")
                print(f"  毒性文本平均分: {toxic_attack:.4f}")
                print(f"  非毒性文本平均分: {nontoxic_attack:.4f}")
                print(f"  区分度 Δ: {toxic_attack - nontoxic_attack:+.4f}")

            if positive_adjs:
                toxic_positive = concept_matrix[toxic_mask][:, positive_adjs].mean()
                nontoxic_positive = concept_matrix[non_toxic_mask][:, positive_adjs].mean()
                print(f"\n语义一致性（正面形容词）:")
                print(f"  毒性文本平均分: {toxic_positive:.4f}")
                print(f"  非毒性文本平均分: {nontoxic_positive:.4f}")
                print(f"  区分度 Δ: {nontoxic_positive - toxic_positive:+.4f}")

    # 5. 示例
    print(f"\n概念向量示例 (前3个样本的Top-5形容词):")
    for i in range(min(3, N)):
        top5_idx = np.argsort(concept_matrix[i])[::-1][:5]
        top5 = [(adj_names[j], concept_matrix[i, j]) for j in top5_idx]
        toxic = data[i].get("toxic", "?") if i < len(data) else "?"
        print(f"  样本{i} (toxic={toxic}): {top5}")

    print(f"{'=' * 60}")


# =============================================================================
# 核心流程
# =============================================================================
def generate_adj_concept_embedding(
    data_path: Path, output_path: Path, csv_output_path: Path,
    adjective_path: Path, tokenizer, model,
    batch_size: int = 64, max_length: int = 512,
    save_embedding: bool = False,
):
    """生成形容词概念向量（Embedding 映射版）。"""
    # --- 准备工作 ---
    # 加载形容词词典
    adj_df = pd.read_csv(adjective_path)
    adj_names = adj_df["chinese"].tolist()
    num_adjs = len(adj_names)

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    print(f"形容词数: {num_adjs}")
    print(f"样本数: {len(data_set)}")

    # --- Step 1: 编码形容词 ---
    print(f"\n{'=' * 60}")
    print("Step 1: 编码形容词")
    print(f"{'=' * 60}")
    adj_texts = build_adj_texts(adj_df)
    for i, t in enumerate(adj_texts[:3]):
        print(f"  [{i}] {t[:60]}...")
    print(f"  ... 共 {len(adj_texts)} 条")

    h_adj = encode_texts(adj_texts, tokenizer, model, batch_size=batch_size,
                         max_length=max_length, desc="Encoding adjectives")
    print(f"  h_adj shape: {h_adj.shape}")

    # --- Step 2: 编码文本 ---
    print(f"\n{'=' * 60}")
    print("Step 2: 编码文本")
    print(f"{'=' * 60}")
    text_contents = [sample["content"] for sample in data_set]
    h_text = encode_texts(text_contents, tokenizer, model, batch_size=batch_size,
                          max_length=max_length, desc="Encoding texts")
    print(f"  h_text shape: {h_text.shape}")

    # --- Step 3: 计算概念向量 ---
    print(f"\n{'=' * 60}")
    print("Step 3: 计算 cosine similarity 概念向量")
    print(f"{'=' * 60}")
    concept_matrix = compute_concept_matrix(h_text, h_adj)
    print(f"  concept_matrix shape: {concept_matrix.shape}")
    print(f"  值域: [{concept_matrix.min():.4f}, {concept_matrix.max():.4f}]")

    # --- Step 4: 质量评估 ---
    evaluate_concept_quality(concept_matrix, adj_names, data_set)

    # --- Step 5: 保存结果 ---
    print(f"\n{'=' * 60}")
    print("Step 5: 保存结果")
    print(f"{'=' * 60}")

    # JSON：含完整信息
    results = []
    for i, sample in enumerate(data_set):
        results.append({
            "content": sample["content"],
            "toxic": sample["toxic"],
            "concept": [round(float(v), 6) for v in concept_matrix[i]],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"  JSON: {output_path}")

    # CSV：纯矩阵
    df = pd.DataFrame(concept_matrix, columns=adj_names)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"  CSV: {csv_output_path}")

    # Embedding 矩阵（可选，用于调试和复现）
    if save_embedding:
        emb_dir = output_path.parent / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)

        np.save(emb_dir / f"h_text_{output_path.stem}.npy", h_text)
        np.save(emb_dir / f"h_adj_{output_path.stem}.npy", h_adj)
        print(f"  h_text embedding: {emb_dir / f'h_text_{output_path.stem}.npy'}")
        print(f"  h_adj embedding: {emb_dir / f'h_adj_{output_path.stem}.npy'}")

    # 评分分布统计
    # 将 [0,1] 映射回近似Likert评分便于理解
    approx_ratings = np.round(concept_matrix * 4 + 1).astype(int)
    approx_ratings = np.clip(approx_ratings, 1, 5)
    rating_counts = {i: int((approx_ratings == i).sum()) for i in range(1, 6)}
    total = approx_ratings.size
    print(f"\n近似Likert评分分布（cos→1-5映射，仅供参考）:")
    for r, c in rating_counts.items():
        print(f"  {r}分: {c:,} ({100*c/total:.1f}%)")

    return concept_matrix


# =============================================================================
# 主入口
# =============================================================================
def main():
    args = parse_args()

    # 设置确定性模式
    if not args.no_deterministic:
        set_deterministic_mode()

    # 构建路径
    data_path = TOXICN_DIR / f"{args.mode}.json"
    adjective_path = ADJ_DIR / args.adjective_name

    if not data_path.exists():
        raise FileNotFoundError(f"数据集不存在: {data_path}")
    if not adjective_path.exists():
        raise FileNotFoundError(f"形容词词典不存在: {adjective_path}")

    # 从词典文件名提取词干
    adj_stem = adjective_path.stem
    adj_version = adj_stem.replace("toxic_adjectives_", "")

    # 输出目录
    concept_dir = OUTPUT_DIR / f"{args.model_name}_embedding"
    concept_dir.mkdir(parents=True, exist_ok=True)

    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{adj_version}.json"
    csv_output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{adj_version}.csv"

    # 打印配置
    print("\n" + "=" * 60)
    print("形容词概念向量生成(Embedding映射版) - 配置信息")
    print("=" * 60)
    print(f"LLM模型名称: {args.model_name}")
    print(f"形容词词典: {adjective_path.name}")
    print(f"当前模式: {args.mode}")
    print(f"数据集路径: {data_path}")
    print(f"JSON输出路径: {output_path}")
    print(f"CSV输出路径: {csv_output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"确定性模式: {'禁用' if args.no_deterministic else '启用'}")
    print(f"保存Embedding: {'是' if args.save_embedding else '否'}")
    print("=" * 60 + "\n")

    # 加载模型
    tokenizer, model, model_config = load_encoder_model(
        MODELS_PATH, args.model_name, args.gpu_memory_utilization
    )

    # 可复现性校验
    if not args.no_deterministic:
        sample_texts = ["这是一条测试文本", "什么被害妄想猎巫man", "你好世界"]
        verify_reproducibility(sample_texts, tokenizer, model)

    # 执行概念向量生成
    max_length = model_config.get("max_length", 512)

    generate_adj_concept_embedding(
        data_path, output_path, csv_output_path, adjective_path,
        tokenizer, model,
        batch_size=args.batch_size,
        max_length=max_length,
        save_embedding=args.save_embedding,
    )

    # 释放 GPU 显存
    del model
    torch.cuda.empty_cache()

    print("\n生成完成！")


if __name__ == '__main__':
    main()
