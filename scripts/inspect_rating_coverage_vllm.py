"""LLM直接评估评分分布全景分析工具（全形容词扫描，vLLM版本）

【定位】
本脚本是 generate_adjective_c_r_vllm.py 的"全形容词切片"评估工具。
generate_adjective_c_r_vllm.py 负责为数据集中所有文本、所有形容词批量生成概念向量；
inspect_prompt_template_vllm.py 负责在单样本级别（一个文本 + 一个形容词）调试提示词和输出格式；
而本脚本则对"一条固定文本 + 全部形容词"进行扫描，评估该提示词在整个形容词词典上的
评分分布和区分度是否合理。

【核心功能】
对单条文本遍历所有形容词，使用 vLLM 推理并提取 LLM 直接输出的评分。

重点分析：
1. 评分分布统计
   - 1-5各评分的频次和占比，判断是否存在评分坍缩（如全部给1分或3分）
   - 理想分布：大量1分（不相关），少量高分区（强相关），中间适度
2. 概念向量区分度
   - 非零分数的比例（区分度指标）：过高说明评分过于宽松，过低说明过于保守
   - 分数标准差：衡量概念向量中各维度之间的区分程度
3. 首 token 概率分析
   - 数字token概率总和：评估提示词对输出格式的约束是否稳定
   - 概率最高的数字与LLM实际输出评分的一致性

【与 generate_adjective_c_r_vllm.py / inspect_prompt_template_vllm.py 的关系】
- 本脚本的提示词构建逻辑、评分解析逻辑与 generate_adjective_c_r_vllm.py 完全一致。
- inspect_prompt_template_vllm.py 用于"点"级别的单样本调试（快速迭代提示词）；
- 本脚本用于"面"级别的全景验证（确认改进后的提示词在整个形容词词典上评分分布合理）；
- 两者结合，确保 generate_adjective_c_r_vllm.py 批量生成的概念向量质量可靠。

【输出】
1. 可视化图表（PNG）：
   - 横轴为形容词索引，纵轴为评分/概率值
   - 包含：LLM评分折线、概念向量分数折线、数字token概率总和折线
2. JSON 数据文件：每个形容词的详细数据 + 统计摘要
3. 控制台输出：评分分布统计和区分度指标

【使用方法】
1. 修改下方 CONFIG 区域的变量（模型名、文本内容等）
2. 运行：python scripts/inspect_rating_coverage_vllm.py
"""
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ADJ_DIR = DATA_DIR / "adjective"
MODELS_PATH = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output" / "experiments" / "rating_coverage"

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== CONFIG 区域（直接修改以下变量）====================
MODEL_NAME = "Qwen2.5-7B-Instruct"  # models目录下的模型文件夹名

# 文本内容（直接修改即可）
TEXT_CONTENT = "什么被害妄想猎巫man"

# 形容词词典文件名
ADJECTIVE_NAME = "toxic_adjectives_v1.csv"

# vLLM推理配置
GPU_MEMORY_UTILIZATION = 0.85  # GPU显存占用比例（0.0-1.0）
# ===================================================================


# 模型加载配置表（与 generate_adjective_c_r_vllm.py 保持一致）
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen3.5-9B": {
        "quantization": "fp8",
        "is_qwen3": True,
        "is_multimodal": True,
        "prompt_suffix": "",
    },
    "glm-4-9b-chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "\n",
    },
    "deepseek-llm-7b-chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Baichuan2-7B-Chat": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen3-8B": {
        "quantization": None,
        "is_qwen3": True,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
}


def get_model_loading_config(model_name: str) -> dict:
    """从 MODEL_LOADING_CONFIG 中获取模型加载配置。未知模型将直接报错。"""
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(
            f"不支持的模型: {model_name}。"
            f"请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    """加载vLLM模型和tokenizer（复用generate_adjective_c_r_vllm逻辑）

    所有模型差异（量化方式、多模态处理、Qwen3+ 标志）均从
    MODEL_LOADING_CONFIG 中读取，保证新增模型时只需改配置表。
    """
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = get_model_loading_config(model_name)
    quantization = model_config["quantization"]
    is_multimodal = model_config["is_multimodal"]

    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=str(llm_path),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,    # 启用前缀缓存，加速system指令复用
        max_model_len=1024,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
    )
    if quantization is not None:
        llm_kwargs["quantization"] = quantization

    # 多模态模型：纯文本推理时跳过视觉编码器，释放显存
    if is_multimodal:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm_kwargs["language_model_only"] = True
        print(f"检测到多模态模型({model_name})，已设置limit_mm_per_prompt + language_model_only")

    print(f"Loading vLLM model from {llm_path}")
    print(f"  量化方式: {quantization if quantization else '无量化'}")
    llm = LLM(**llm_kwargs)

    return tokenizer, llm, model_config["is_qwen3"]


def get_first_token_ids(word_list, tokenizer):
    """获取词表中每个词的首token id（去重）"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])

    if not token_ids:
        raise ValueError("get_first_token_ids ERROR: 词表中无有效token")

    return list(dict.fromkeys(token_ids))


# system instruction（与generate_adjective_c_r_vllm.py一致）
SYSTEM_INSTRUCTION = (
    "你是一位语言分析专家，擅长识别文本中的隐含语义。\n"
    "在评估时，请综合考虑文本的字面意思和可能的隐含意思（如比喻、谐音、反讽、文化隐喻等），\n"
    "判断文本是否体现了该形容词所描述的特征。\n"
    "\n"
    "评分标准（1-5分）：\n"
    "1 - 完全不相关：文本与该形容词描述的特征毫无关联\n"
    "2 - 微弱相关：文本仅隐约或间接体现该特征\n"
    "3 - 部分相关：文本在一定程度上体现该特征\n"
    "4 - 较为相关：文本明显体现该特征\n"
    "5 - 高度相关：文本是该特征的典型表现\n"
    "\n"
    "注意事项：\n"
    "- 对行为的正当批评（如谴责暴力）不等于该行为本身\n"
    "- 宁可低估也不要高估，拿不准的倾向给低分\n"
    "- 严格按指定格式输出"
)


def build_chat_messages(content, adj, adj_definition=None):
    """
    构建直接评估Chat Template的messages列表。
    逻辑与 generate_adjective_c_r_vllm.py 中的模板构建保持一致。
    """
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"该文本在多大程度上体现了\"{adj}\"所描述的特征？")
    user_lines.append("请评分（1-5）：")
    user_content = "\n".join(user_lines)

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]
    return messages


def parse_rating(output_text: str) -> int:
    """从LLM输出文本中解析1~5的评分（与generate_adjective_c_r_vllm.py一致）。

    解析策略（按优先级）：
    1. 提取"评分"关键字后的数字
    2. 提取首个独立数字（1-5）
    3. 中文数字映射
    4. 找不到则返回1（保守默认值）
    """
    # 方法1：找"评分"关键字后的数字
    rating_match = re.search(r'评分[是为：:]\s*([1-5])', output_text)
    if rating_match:
        return int(rating_match.group(1))

    # 方法2：提取首个独立的1-5数字
    match = re.search(r'\b([1-5])\b', output_text)
    if match:
        return int(match.group(1))

    # 方法3：中文数字映射
    cn_num_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    for cn_char, num in cn_num_map.items():
        if cn_char in output_text:
            return num

    # 解析失败，返回1（保守默认）
    return 1


def rating_to_score(rating: int) -> float:
    """将1~5评分归一化到[0, 1]（与generate_adjective_c_r_vllm.py一致）。"""
    return (rating - 1) / 4.0


def analyze_rating_coverage(
    text_content,
    adjective_path,
    tokenizer,
    llm_model,
    output_dir: Path,
    model_name: str,
    is_qwen3=False,
    prompt_suffix="",
):
    """
    对单条文本遍历所有形容词，使用 vLLM 推理并提取 LLM 直接输出的评分，
    分析评分分布和概念向量区分度。
    """
    # 定义数字token（用于首token概率分析）
    rating_tokens = ["1", "2", "3", "4", "5"]
    rating_ids = get_first_token_ids(rating_tokens, tokenizer)

    # 加载形容词词典（含定义）
    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    adj_en_list = adj_df["adjective"].tolist() if "adjective" in adj_df.columns else [""] * len(adjectives)
    adj_definitions = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adjectives)

    # vLLM采样配置：同时获取生成文本和首token logprobs
    sampling_params = SamplingParams(
        max_tokens=32,
        temperature=0,
        logprobs=20
    )

    # 存储结果
    results = []

    # 构建所有提示词
    prompts = []
    for adj, adj_def in zip(adjectives, adj_definitions):
        messages = build_chat_messages(text_content, adj, adj_def)

        chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs
        )
        # 追加模型特定的后缀
        prompt_text += prompt_suffix
        prompts.append(prompt_text)

    # 批量推理（vLLM自动处理批量化）
    outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

    parse_fail_count = 0

    for adj_idx, sample_info in enumerate(tqdm(outputs, desc="Processing adjectives")):
        # 提取LLM生成的文本
        generated_text = sample_info.outputs[0].text.strip()

        # 解析评分
        rating = parse_rating(generated_text)
        score = rating_to_score(rating)

        # 检测解析失败
        if rating == 1 and "1" not in generated_text[:5]:
            parse_fail_count += 1

        # 提取首token的logprobs（数字token概率分析）
        logprobs = sample_info.outputs[0].logprobs
        first_token_logprobs = logprobs[0] if logprobs else {}

        probs_dict = {}
        for token_id, logprob_obj in first_token_logprobs.items():
            probs_dict[token_id] = math.exp(logprob_obj.logprob)

        # 数字token概率
        level_probs = [probs_dict.get(tid, 0.0) for tid in rating_ids]
        total_digit_prob = sum(level_probs)

        # 概率最高的数字
        max_prob_idx = level_probs.index(max(level_probs))
        max_prob_rating = max_prob_idx + 1  # 1-5

        # 首token与LLM输出是否一致
        first_token_match = (max_prob_rating == rating)

        results.append({
            "index": adj_idx,
            "adjective_en": adj_en_list[adj_idx],
            "adjective_cn": adjectives[adj_idx],
            "generated_text": generated_text,
            "rating": rating,
            "score": round(score, 4),
            "level_1_prob": round(level_probs[0], 6),
            "level_2_prob": round(level_probs[1], 6),
            "level_3_prob": round(level_probs[2], 6),
            "level_4_prob": round(level_probs[3], 6),
            "level_5_prob": round(level_probs[4], 6),
            "total_digit_prob": round(total_digit_prob, 6),
            "max_prob_rating": max_prob_rating,
            "first_token_match": first_token_match,
        })

    # ---- 统计分析 ----
    all_ratings = [r["rating"] for r in results]
    all_scores = [r["score"] for r in results]
    total_digit_probs = [r["total_digit_prob"] for r in results]

    rating_counts = {i: all_ratings.count(i) for i in range(1, 6)}
    total = len(all_ratings)
    nonzero_ratio = sum(1 for s in all_scores if s > 0) / total  # 非零分数比例
    score_std = (sum((s - sum(all_scores) / total) ** 2 for s in all_scores) / total) ** 0.5
    first_token_match_ratio = sum(1 for r in results if r["first_token_match"]) / total
    avg_digit_prob = sum(total_digit_probs) / len(total_digit_probs)

    # ---- 保存JSON数据 ----
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_text = text_content[:20].replace("\\", "").replace("/", "").replace(" ", "_")
    json_path = output_dir / f"direct_{safe_text}_{model_name}_vllm.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "text_content": text_content,
            "num_adjectives": len(adjectives),
            "statistics": {
                "rating_distribution": {str(k): v for k, v in rating_counts.items()},
                "nonzero_score_ratio": round(nonzero_ratio, 4),
                "score_std": round(score_std, 4),
                "score_mean": round(sum(all_scores) / total, 4),
                "parse_fail_count": parse_fail_count,
                "first_token_match_ratio": round(first_token_match_ratio, 4),
                "mean_digit_prob": round(avg_digit_prob, 6),
                "min_digit_prob": round(min(total_digit_probs), 6),
                "max_digit_prob": round(max(total_digit_probs), 6),
            },
            "data": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"数据已保存: {json_path}")

    # ---- 绘制图表 ----
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    x = [r["index"] for r in results]
    scores = [r["score"] for r in results]
    ratings_list = [r["rating"] for r in results]

    # 上图：LLM评分和概念向量分数
    ax1 = axes[0]
    ax1.plot(x, ratings_list, label="LLM评分 (1-5)", color="orange", alpha=0.9, linewidth=1.2, marker="o", markersize=2)
    ax1.plot(x, scores, label="概念向量分数 [0,1]", color="green", alpha=0.8, linewidth=1.0, linestyle="-.", marker="s", markersize=2)
    mean_score = sum(scores) / len(scores)
    ax1.axhline(y=mean_score, color="green", linestyle="--", alpha=0.5, label=f"分数均值: {mean_score:.3f}")
    ax1.set_ylabel("评分 / 分数", fontsize=12)
    ax1.set_title(
        f"LLM直接评估评分分布分析\n模型: {model_name} | 文本: {text_content[:30]}...",
        fontsize=14,
    )
    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_ylim(-0.1, 5.5)
    ax1.grid(True, alpha=0.3)

    # 下图：数字token概率分析
    ax2 = axes[1]
    ax2.plot(x, total_digit_probs, label="数字token概率总和", color="blue", alpha=0.9, linewidth=1.2)
    ax2.axhline(y=avg_digit_prob, color="blue", linestyle="--", alpha=0.5, label=f"概率均值: {avg_digit_prob:.3f}")
    ax2.set_xlabel("形容词索引", fontsize=12)
    ax2.set_ylabel("概率", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 在底部添加形容词名称（稀疏显示，避免重叠）
    tick_step = max(1, len(adjectives) // 20)
    tick_positions = list(range(0, len(adjectives), tick_step))
    tick_labels = [adjectives[i] if i < len(adjectives) else "" for i in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    png_path = output_dir / f"direct_{safe_text}_{model_name}_vllm.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {png_path}")
    plt.close()

    # ---- 打印统计摘要 ----
    print("\n" + "=" * 60)
    print("LLM直接评估评分分布统计摘要")
    print("=" * 60)
    print(f"形容词数量: {len(adjectives)}")
    print(f"\n评分分布:")
    for r, c in rating_counts.items():
        bar = "█" * int(50 * c / total) if total > 0 else ""
        print(f"  {r}分: {c:>4} ({100*c/total:>5.1f}%) {bar}")
    print(f"\n概念向量指标:")
    print(f"  非零分数比例: {nonzero_ratio:.2%}")
    print(f"  分数均值: {sum(all_scores)/total:.4f}")
    print(f"  分数标准差: {score_std:.4f}")
    print(f"\n输出格式约束:")
    print(f"  数字token概率均值: {avg_digit_prob:.4f}")
    print(f"  数字token概率范围: [{min(total_digit_probs):.4f}, {max(total_digit_probs):.4f}]")
    print(f"  首 token 与输出一致性: {first_token_match_ratio:.2%}")
    if parse_fail_count > 0:
        print(f"  解析失败(默认1分): {parse_fail_count}")

    # 诊断建议
    print(f"\n{'=' * 60}")
    print("诊断建议:")
    if rating_counts.get(1, 0) / total > 0.9:
        print("  ⚠ 评分严重坍缩至1分(>90%)，概念向量区分度极低")
        print("    可能原因：提示词过于保守、模型能力不足、文本确实与多数形容词不相关")
    elif rating_counts.get(1, 0) / total > 0.7:
        print("  ⚠ 评分偏向1分(>70%)，区分度偏低")
    else:
        print("  ✓ 评分分布有一定区分度")

    if nonzero_ratio < 0.1:
        print("  ⚠ 非零分数比例极低(<10%)，概念向量过于稀疏")
    elif nonzero_ratio > 0.5:
        print("  ⚠ 非零分数比例偏高(>50%)，评分可能过于宽松")
    else:
        print("  ✓ 非零分数比例适中")

    if avg_digit_prob < 0.5:
        print("  ⚠ 数字token概率均值过低(<50%)，提示词格式约束不稳定")
    else:
        print("  ✓ 数字token概率均值良好，格式约束稳定")

    if first_token_match_ratio < 0.7:
        print("  ⚠ 首 token 与输出一致性低(<70%)，LLM可能在首 token 后改变方向")
    else:
        print("  ✓ 首 token 与输出一致性高")

    print("=" * 60)

    return results


def main():
    adjective_path = ADJ_DIR / ADJECTIVE_NAME
    if not adjective_path.exists():
        raise FileNotFoundError(f"形容词词典不存在: {adjective_path}")

    print("\n" + "=" * 60)
    print("LLM直接评估评分分布分析（vLLM版本）")
    print("=" * 60)
    print(f"模型名称: {MODEL_NAME}")
    print(f"文本内容: {TEXT_CONTENT}")
    print(f"形容词词典: {ADJECTIVE_NAME}")
    print(f"GPU显存占用: {GPU_MEMORY_UTILIZATION}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    tokenizer, llm_model, qwen3_flag = load_vllm_model(MODELS_PATH, MODEL_NAME, GPU_MEMORY_UTILIZATION)
    if qwen3_flag:
        print(f"检测到Qwen3+模型({MODEL_NAME})，已禁用思考模式(enable_thinking=False)")
    model_config = get_model_loading_config(MODEL_NAME)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"检测到模型({MODEL_NAME})需要追加prompt后缀: {repr(prompt_suffix)}")

    analyze_rating_coverage(
        text_content=TEXT_CONTENT,
        adjective_path=adjective_path,
        tokenizer=tokenizer,
        llm_model=llm_model,
        output_dir=Path(OUTPUT_DIR),
        model_name=MODEL_NAME,
        is_qwen3=qwen3_flag,
        prompt_suffix=prompt_suffix,
    )


if __name__ == "__main__":
    main()
