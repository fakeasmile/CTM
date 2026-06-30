"""LLM Embedding 语义空间诊断工具

【定位】
本脚本是 generate_adjective_c_r_embedding.py 的诊断工具。
generate_adjective_c_r_embedding.py 负责通过 LLM Embedding 生成概念向量；
而本脚本用于在批量生成前/后验证 Embedding 空间的语义质量，
确保 LLM 编码的语义向量确实能区分不同形容词、对毒性文本有正确的语义响应。

【核心功能】
1. 形容词 Embedding 区分度分析
   - 计算形容词之间的 cosine similarity 矩阵
   - 语义相近的形容词应有高相似度，语义相反的应有低相似度
   - 如果所有形容词的 embedding 都很相似，说明空间区分度不足

2. 可复现性校验
   - 对同一文本编码两次，验证结果是否一致
   - 确保编码模式不存在随机性

3. 文本-形容词语义对齐验证
   - 用几条典型文本（毒性/非毒性）验证概念向量的合理性
   - 毒性文本应对"攻击性""辱骂性"等形容词有较高 concept 值

4. 可视化
   - t-SNE/PCA 降维可视化形容词在语义空间中的分布
   - 形容词间 cosine similarity 热力图

【与 generate_adjective_c_r_embedding.py 的关系】
- 共享相同的模型加载和编码逻辑
- 用于在批量生成前快速验证 Embedding 方案是否可行
- 也可用于分析已保存的 Embedding 矩阵

【使用方法】
1. 直接运行（使用 CONFIG 区域的配置）：
   python scripts/inspect_embedding_space.py

2. 分析已保存的 Embedding：
   python scripts/inspect_embedding_space.py --load_embedding
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ADJ_DIR = DATA_DIR / "adjective"
TOXICN_DIR = DATA_DIR / "TOXICN"
OUTPUT_DIR = PROJECT_ROOT / "output" / "experiments"
MODELS_PATH = PROJECT_ROOT / "models"

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ==================== CONFIG 区域 ====================
MODEL_NAME = "Qwen2.5-7B-Instruct"

# 测试文本（用于语义对齐验证）
TEST_TEXTS = [
    ("毒性文本-辱骂", "你个傻逼废物滚出去"),
    ("毒性文本-威胁", "信不信我找人弄死你"),
    ("毒性文本-歧视", "那些外地人就是来抢我们资源的"),
    ("非毒性-理性讨论", "我认为这个政策还需要进一步完善"),
    ("非毒性-赞扬", "你的想法很有创意，值得借鉴"),
    ("非毒性-调解", "大家冷静一下，先听对方说完"),
]

# 输出目录
OUTPUT_SUBDIR = "embedding_diagnostics"

# GPU显存占用
GPU_MEMORY_UTILIZATION = 0.9
# ====================================================


# 模型加载配置表（与 generate_adjective_c_r_embedding.py 一致）
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
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(f"不支持的模型: {model_name}")
    return MODEL_LOADING_CONFIG[model_name].copy()


def load_encoder_model(model_path: Path, model_name: str):
    """加载编码器模型（与 generate_adjective_c_r_embedding.py 一致）"""
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = get_model_loading_config(model_name)

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {llm_path}")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32,
        "output_hidden_states": True,
    }
    quantization = model_config["quantization"]
    if quantization is not None:
        from transformers import BitsAndBytesConfig
        if quantization == "fp8":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModel.from_pretrained(llm_path, **model_kwargs)
    if quantization is None:
        model = model.cuda()
    model.eval()

    return tokenizer, model, model_config


@torch.no_grad()
def encode_texts(texts: list[str], tokenizer, model, batch_size: int = 64,
                 max_length: int = 512, desc: str = "Encoding") -> np.ndarray:
    """批量编码文本（与 generate_adjective_c_r_embedding.py 一致）"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked_hidden = last_hidden * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        count = attention_mask.sum(dim=1).clamp(min=1e-9)
        embeddings = sum_hidden / count
        all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def build_adj_texts(adj_df: pd.DataFrame) -> list[str]:
    """构建形容词编码文本（与 generate_adjective_c_r_embedding.py 一致）"""
    texts = []
    for _, row in adj_df.iterrows():
        adj_cn = row["chinese"]
        definition = row.get("definition", "")
        if isinstance(definition, str) and definition.strip():
            texts.append(f"{adj_cn}：{definition}")
        else:
            texts.append(adj_cn)
    return texts


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """计算 A 和 B 之间的 cosine similarity 矩阵"""
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-9)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-9)
    return A_norm @ B_norm.T


# =============================================================================
# 诊断1：可复现性校验
# =============================================================================
def verify_reproducibility(tokenizer, model):
    """编码两次，验证结果一致"""
    test_texts = ["这是一条测试文本", "什么被害妄想猎巫man", "你好世界"]
    emb1 = encode_texts(test_texts, tokenizer, model, batch_size=3, desc="Repro #1")
    emb2 = encode_texts(test_texts, tokenizer, model, batch_size=3, desc="Repro #2")

    max_diff = np.abs(emb1 - emb2).max()
    mean_diff = np.abs(emb1 - emb2).mean()

    print(f"\n{'=' * 60}")
    print("诊断1：可复现性校验")
    print(f"{'=' * 60}")
    print(f"  最大差异: {max_diff:.2e}")
    print(f"  平均差异: {mean_diff:.2e}")
    if max_diff < 1e-5:
        print("  ✓ 可复现性优秀")
    elif max_diff < 1e-3:
        print("  ⚠ 可复现性可接受，存在微小浮点差异")
    else:
        print("  ✗ 可复现性差，建议启用确定性模式")
    return max_diff


# =============================================================================
# 诊断2：形容词 Embedding 区分度分析
# =============================================================================
def analyze_adj_embedding_quality(h_adj: np.ndarray, adj_names: list[str],
                                  adj_en_names: list[str], output_dir: Path):
    """分析形容词 embedding 的区分度"""
    V = len(adj_names)

    # 计算形容词间 cosine similarity
    cos_sim = cosine_similarity_matrix(h_adj, h_adj)

    # 去掉对角线（自相似度=1）
    mask = ~np.eye(V, dtype=bool)
    off_diag = cos_sim[mask]

    print(f"\n{'=' * 60}")
    print("诊断2：形容词 Embedding 区分度分析")
    print(f"{'=' * 60}")
    print(f"  形容词数量: {V}")
    print(f"  Embedding 维度: {h_adj.shape[1]}")
    print(f"\n  形容词间 Cosine Similarity 统计:")
    print(f"    均值:   {off_diag.mean():.4f}")
    print(f"    中位数: {np.median(off_diag):.4f}")
    print(f"    最小值: {off_diag.min():.4f}")
    print(f"    最大值: {off_diag.max():.4f}")
    print(f"    标准差: {off_diag.std():.4f}")

    # 区分度指标：off-diag 的标准差越大，区分度越好
    if off_diag.std() < 0.02:
        print("  ⚠ 区分度极低：形容词 embedding 几乎相同，空间坍缩")
    elif off_diag.std() < 0.05:
        print("  ⚠ 区分度偏低：形容词 embedding 区分有限")
    else:
        print("  ✓ 区分度良好：形容词 embedding 有明显的语义差异")

    # 语义一致性验证：找几对应该相近/应该远离的形容词
    print(f"\n  语义一致性验证:")

    # 找相近的形容词对
    attack_adj = [i for i, n in enumerate(adj_names) if "攻击" in n]
    abusive_adj = [i for i, n in enumerate(adj_names) if "辱骂" in n]
    if attack_adj and abusive_adj:
        sim = cos_sim[attack_adj[0], abusive_adj[0]]
        print(f"    '攻击性的' vs '辱骂性的': {sim:.4f} (应高)")

    accepting_adj = [i for i, n in enumerate(adj_names) if "包容" in n]
    appreciative_adj = [i for i, n in enumerate(adj_names) if "赞赏" in n]
    if accepting_adj and appreciative_adj:
        sim = cos_sim[accepting_adj[0], appreciative_adj[0]]
        print(f"    '包容的' vs '赞赏的': {sim:.4f} (应较高)")

    if attack_adj and accepting_adj:
        sim = cos_sim[attack_adj[0], accepting_adj[0]]
        print(f"    '攻击性的' vs '包容的': {sim:.4f} (应较低)")

    # 最相似和最不相似的形容词对（排除自身）
    np.fill_diagonal(cos_sim, -2)  # 掩盖对角线
    most_similar_idx = np.unravel_index(cos_sim.argmax(), cos_sim.shape)
    least_similar_idx = np.unravel_index(cos_sim[cos_sim > -2].min() if (cos_sim > -2).any() else 0, cos_sim.shape)

    # 重新计算（恢复对角线）
    cos_sim_restore = cosine_similarity_matrix(h_adj, h_adj)
    np.fill_diagonal(cos_sim_restore, 0)

    # Top-5 最相似对
    triu_idx = np.triu_indices(V, k=1)
    triu_vals = cos_sim_restore[triu_idx]
    top5_sim = np.argsort(triu_vals)[::-1][:5]
    print(f"\n  最相似的5对形容词:")
    for rank, idx in enumerate(top5_sim, 1):
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        print(f"    {rank}. {adj_names[i]} vs {adj_names[j]}: {triu_vals[idx]:.4f}")

    # Top-5 最不相似对
    bot5_sim = np.argsort(triu_vals)[:5]
    print(f"\n  最不相似的5对形容词:")
    for rank, idx in enumerate(bot5_sim, 1):
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        print(f"    {rank}. {adj_names[i]} vs {adj_names[j]}: {triu_vals[idx]:.4f}")

    # 保存 cosine similarity 矩阵
    output_dir.mkdir(parents=True, exist_ok=True)
    cos_df = pd.DataFrame(cos_sim_restore, index=adj_names, columns=adj_names)
    cos_df.to_csv(output_dir / f"adj_cosine_sim_{MODEL_NAME}.csv", encoding="utf-8-sig")
    print(f"\n  Cosine similarity 矩阵已保存: {output_dir / f'adj_cosine_sim_{MODEL_NAME}.csv'}")

    return cos_sim_restore


# =============================================================================
# 诊断3：文本-形容词语义对齐验证
# =============================================================================
def verify_text_adj_alignment(h_adj: np.ndarray, adj_names: list[str],
                              tokenizer, model, test_texts: list[tuple]):
    """验证概念向量对典型文本的语义合理性"""
    print(f"\n{'=' * 60}")
    print("诊断3：文本-形容词语义对齐验证")
    print(f"{'=' * 60}")

    for label, text in test_texts:
        # 编码单条文本
        h_text = encode_texts([text], tokenizer, model, batch_size=1, desc=f"Testing '{label}'")

        # 计算概念向量
        concept = compute_concept_vector(h_text[0], h_adj)

        # 显示 Top-5 和 Bottom-5 形容词
        top5_idx = np.argsort(concept)[::-1][:5]
        bot5_idx = np.argsort(concept)[:5]

        print(f"\n  [{label}] 文本: \"{text}\"")
        print(f"    Top-5 相关形容词:")
        for i, idx in enumerate(top5_idx, 1):
            print(f"      {i}. {adj_names[idx]}: {concept[idx]:.4f}")
        print(f"    Bottom-5 相关形容词:")
        for i, idx in enumerate(bot5_idx, 1):
            print(f"      {i}. {adj_names[idx]}: {concept[idx]:.4f}")

    # 综合判断
    print(f"\n  语义对齐判断:")
    print(f"    如果毒性文本的Top-5中出现'攻击性''辱骂性'等，非毒性文本出现'包容的''赞赏的'等，则对齐良好")


def compute_concept_vector(h_text: np.ndarray, h_adj: np.ndarray) -> np.ndarray:
    """计算单条文本的概念向量"""
    h_text_norm = h_text / max(np.linalg.norm(h_text), 1e-9)
    h_adj_norm = h_adj / np.linalg.norm(h_adj, axis=1, keepdims=True).clip(min=1e-9)
    cos_vals = h_adj_norm @ h_text_norm
    cos_vals = np.clip(cos_vals, -1.0, 1.0)
    return (cos_vals + 1.0) / 2.0  # 映射到 [0, 1]


# =============================================================================
# 诊断4：降维可视化
# =============================================================================
def visualize_embeddings(h_adj: np.ndarray, adj_names: list[str],
                         output_dir: Path, model_name: str):
    """t-SNE 降维可视化"""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("  跳过可视化（缺少 sklearn 或 matplotlib）")
        return

    print(f"\n{'=' * 60}")
    print("诊断4：降维可视化")
    print(f"{'=' * 60}")

    # t-SNE 降维
    print("  运行 t-SNE 降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(adj_names) - 1))
    h_2d = tsne.fit_transform(h_adj)

    # 绘图
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(h_2d[:, 0], h_2d[:, 1], s=30, alpha=0.7)

    # 标注形容词名称
    for i, name in enumerate(adj_names):
        ax.annotate(name, (h_2d[i, 0], h_2d[i, 1]),
                   fontsize=6, alpha=0.8,
                   xytext=(3, 3), textcoords='offset points')

    ax.set_title(f"形容词 Embedding t-SNE 可视化\n模型: {model_name}", fontsize=14)
    ax.set_xlabel("t-SNE 维度1", fontsize=12)
    ax.set_ylabel("t-SNE 维度2", fontsize=12)
    ax.grid(True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"adj_tsne_{model_name}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  图表已保存: {png_path}")
    plt.close()

    # Cosine similarity 热力图
    cos_sim = cosine_similarity_matrix(h_adj, h_adj)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cos_sim, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    # 稀疏标注
    tick_step = max(1, len(adj_names) // 20)
    tick_positions = list(range(0, len(adj_names), tick_step))
    tick_labels = [adj_names[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=7)

    ax.set_title(f"形容词间 Cosine Similarity 热力图\n模型: {model_name}", fontsize=14)
    plt.tight_layout()
    heatmap_path = output_dir / f"adj_cosine_heatmap_{model_name}.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"  热力图已保存: {heatmap_path}")
    plt.close()


# =============================================================================
# 主入口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Embedding 空间诊断工具")
    parser.add_argument('--model_name', type=str, default=MODEL_NAME)
    parser.add_argument('--load_embedding', type=str, default=None,
                        help='加载已保存的 embedding 文件路径前缀（不含 _text/_adj 后缀）')
    parser.add_argument('--skip_visualization', action='store_true', help='跳过可视化')
    args = parser.parse_args()

    model_name = args.model_name

    # 加载形容词词典
    adj_csv = ADJ_DIR / "toxic_adjectives_v1.csv"
    adj_df = pd.read_csv(adj_csv)
    adj_names = adj_df["chinese"].tolist()
    adj_en_names = adj_df["adjective"].tolist() if "adjective" in adj_df.columns else [""] * len(adj_names)

    output_dir = OUTPUT_DIR / OUTPUT_SUBDIR

    if args.load_embedding:
        # 从已保存的文件加载
        print(f"从文件加载 Embedding: {args.load_embedding}")
        h_adj = np.load(f"{args.load_embedding}_adj.npy")
        h_text = np.load(f"{args.load_embedding}_text.npy")
        print(f"  h_adj: {h_adj.shape}, h_text: {h_text.shape}")
        tokenizer, model = None, None
    else:
        # 加载模型并编码
        print("\n" + "=" * 60)
        print("Embedding 空间诊断工具")
        print("=" * 60)
        print(f"模型: {model_name}")
        print(f"形容词词典: {adj_csv.name}")
        print("=" * 60 + "\n")

        # 设置确定性模式
        try:
            torch.use_deterministic_mode(True)
            print("已启用确定性模式")
        except Exception as e:
            print(f"警告：确定性模式启用失败: {e}")

        tokenizer, model, model_config = load_encoder_model(MODELS_PATH, model_name)

        # 编码形容词
        adj_texts = build_adj_texts(adj_df)
        h_adj = encode_texts(adj_texts, tokenizer, model, batch_size=32,
                            max_length=model_config.get("max_length", 512),
                            desc="Encoding adjectives")

        # 诊断1：可复现性
        verify_reproducibility(tokenizer, model)

    # 诊断2：形容词区分度
    analyze_adj_embedding_quality(h_adj, adj_names, adj_en_names, output_dir)

    # 诊断3：文本-形容词语义对齐
    if tokenizer is not None and model is not None:
        verify_text_adj_alignment(h_adj, adj_names, tokenizer, model, TEST_TEXTS)

    # 诊断4：可视化
    if not args.skip_visualization:
        visualize_embeddings(h_adj, adj_names, output_dir, model_name)

    # 释放 GPU
    if model is not None:
        del model
        torch.cuda.empty_cache()

    print("\n诊断完成！")


if __name__ == "__main__":
    main()
