"""LLM直接评估提示词模板调试工具（单样本切片分析，vLLM版本）

【定位】
本脚本是 generate_adjective_c_r_vllm.py 的"单样本切片"调试工具。
generate_adjective_c_r_vllm.py 负责为数据集中所有文本、所有形容词批量生成概念向量；
而本脚本只抽取"一个文本 + 一个形容词"进行单步推理，用于在批量生成前快速验证
提示词模板的设计是否合理、LLM是否按预期输出1-5评分。

【核心功能】
1. 首 token 概率分布 Top-10
   观察模型在第一个输出位置的概率分布。如果 Top-10 中包含1-5的数字token，
   说明提示词模板成功将模型输出约束到预期方向。
2. 模型实际生成序列（贪心解码，max_tokens=32）
   观察模型实际输出的文本是否通顺、是否符合模板要求（是否直接回答1-5数字）。
   重点检查：是否输出多余的解释性文字、是否输出非预期的格式。
3. 数字token概率分析
   统计1-5数字token的概率总和和分布，评估提示词对输出格式的约束强度。
   - 理想情况下，1-5数字token应占首token概率质量的较高比例。
   - 若过低，说明模型大量概率分散到非预期token，提示词需改进。
4. 评分解析验证
   使用与 generate_adjective_c_r_vllm.py 完全一致的 parse_rating 函数解析输出，
   验证解析逻辑是否正确提取评分。

【与 generate_adjective_c_r_vllm.py 的关系】
- 本脚本的提示词构建逻辑、评分解析逻辑与 generate_adjective_c_r_vllm.py 完全一致。
- 通过本脚本调试确认模板合理后，再运行 generate_adjective_c_r_vllm.py 进行批量生成，
  可确保生成的概念向量质量。

【使用方法】
直接修改下方 CONFIG 区域的变量（模型名、文本内容、形容词、形容词定义等），然后运行：
python scripts/inspect_prompt_template_vllm.py
"""
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
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


# ==================== CONFIG 区域（直接修改以下变量）====================
MODEL_NAME = "Qwen2.5-7B-Instruct"  # models目录下的模型文件夹名

# 文本内容和形容词（直接修改即可）
TEXT_CONTENT = "什么被害妄想猎巫man"
ADJECTIVE = "包容的"

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
    """获取词表中每个词的首token id"""
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


def main():
    # 加载形容词定义
    adj_csv = ADJ_DIR / "toxic_adjectives_v1.csv"
    adj_definition = None
    if adj_csv.exists():
        adj_df = pd.read_csv(adj_csv)
        if "chinese" in adj_df.columns:
            match = adj_df[adj_df["chinese"] == ADJECTIVE]
            if not match.empty and "definition" in adj_df.columns:
                adj_definition = match.iloc[0]["definition"]

    # 构建Chat Template messages
    messages = build_chat_messages(TEXT_CONTENT, ADJECTIVE, adj_definition)

    # 加载模型
    tokenizer, llm_model, qwen3_flag = load_vllm_model(MODELS_PATH, MODEL_NAME, GPU_MEMORY_UTILIZATION)

    # 生成完整prompt文本
    chat_template_kwargs = {"enable_thinking": False} if qwen3_flag else {}
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs
    )
    # 追加模型特定的后缀
    model_config = get_model_loading_config(MODEL_NAME)
    prompt_suffix = model_config.get("prompt_suffix", "")
    prompt += prompt_suffix

    print("\n" + "=" * 60)
    print("LLM直接评估 - 提示词模板调试（vLLM版本）")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"文本内容: {TEXT_CONTENT}")
    print(f"形容词: {ADJECTIVE}")
    print(f"形容词定义: {adj_definition}")
    print(f"GPU显存占用: {GPU_MEMORY_UTILIZATION}")
    print(f"提示词: {prompt}")
    print(f"提示词token数: {len(tokenizer.encode(prompt))}")

    # ---- 分析1：首token概率分布 ----
    sampling_params_analysis = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=20
    )

    outputs = llm_model.generate([prompt], sampling_params_analysis, use_tqdm=False)
    output = outputs[0]

    logprobs = output.outputs[0].logprobs
    first_token_logprobs = logprobs[0]

    probs_dict = {}
    for token_id, logprob_obj in first_token_logprobs.items():
        probs_dict[token_id] = math.exp(logprob_obj.logprob)

    # 输出概率最高的前10个token
    topk = 10
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
    print(f"\n{'=' * 60}")
    print(f"首token概率分布 Top-{topk}:")
    print(f"{'=' * 60}")
    print(f"{'排名':<4} {'Token ID':<10} {'Token文本':<16} {'概率':<12} {'累计概率':<10}")
    cumsum = 0.0
    for rank, (tid, prob) in enumerate(sorted_probs, 1):
        token_text = tokenizer.decode([tid])
        cumsum += prob
        print(f"{rank:<4} {tid:<10} {repr(token_text):<16} {prob:<12.6f} {cumsum:<10.6f}")

    # ---- 分析2：模型实际生成序列 ----
    print(f"\n{'=' * 60}")
    print(f"模型生成序列（贪心解码，max_tokens=32）:")
    print(f"{'=' * 60}")
    sampling_params_gen = SamplingParams(
        max_tokens=32,
        temperature=0,
        logprobs=None
    )
    outputs_gen = llm_model.generate([prompt], sampling_params_gen, use_tqdm=False)
    generated_text = outputs_gen[0].outputs[0].text
    generated_ids = outputs_gen[0].outputs[0].token_ids
    print(f"生成token序列: {generated_ids}")
    print(f"生成token数量: {len(generated_ids)}")
    print(f"生成文本: {repr(generated_text)}")

    # ---- 分析3：数字token概率分析 ----
    rating_tokens = ["1", "2", "3", "4", "5"]
    rating_ids = get_first_token_ids(rating_tokens, tokenizer)

    print(f"\n{'=' * 60}")
    print(f"数字token概率分析 ({len(rating_tokens)}个评分 → {len(rating_ids)}个唯一token):")
    print(f"{'=' * 60}")
    print(f"{'数字':<8} {'Token ID':<10} {'概率':<12} {'归一化概率':<12}")
    rating_prob_list = []
    for word in rating_tokens:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            tid = encoded[0]
            p = probs_dict.get(tid, 0.0)
            rating_prob_list.append((word, tid, p))

    total_rating_prob = sum(p for _, _, p in rating_prob_list)
    for word, tid, p in rating_prob_list:
        norm_p = p / (total_rating_prob + 1e-8)
        print(f"{word:<8} {tid:<10} {p:<12.6f} {norm_p:<12.6f}")

    print(f"\n数字token概率总和: {total_rating_prob:.6f}")
    print(f"数字token占总概率比例: {total_rating_prob:.2%}")

    # ---- 分析4：评分解析验证 ----
    print(f"\n{'=' * 60}")
    print("评分解析验证（与generate_adjective_c_r_vllm.py一致）:")
    print(f"{'=' * 60}")
    parsed_rating = parse_rating(generated_text)
    normalized_score = rating_to_score(parsed_rating)
    print(f"LLM原始输出: {repr(generated_text)}")
    print(f"解析评分: {parsed_rating}")
    print(f"归一化分数: {normalized_score:.4f}")
    print(f"解析是否成功: {'是' if str(parsed_rating) in generated_text[:5] else '否（使用默认值1）'}")

    # 首token对应的Likert评分（参考信息）
    if total_rating_prob > 0:
        weights = [0.0, 0.25, 0.5, 0.75, 1.0]
        level_probs = [p for _, _, p in rating_prob_list]
        total_prob_eps = total_rating_prob + 1e-8
        likert_score = sum(w * p for w, p in zip(weights, level_probs)) / total_prob_eps
        print(f"\n[参考] 首 token Likert 加权期望分数: {likert_score:.4f}")

    print(f"\n{'=' * 60}")
    print("调试建议:")
    if total_rating_prob < 0.5:
        print("  ⚠ 数字token概率总和过低(<50%)，提示词对输出格式的约束不足")
        print("    建议：加强提示词中对输出格式的约束，例如'只回答一个数字'")
    elif total_rating_prob < 0.7:
        print("  ⚠ 数字token概率总和偏低(50%-70%)，提示词约束可进一步强化")
    else:
        print("  ✓ 数字token概率总和良好(>70%)，提示词对输出格式约束有效")

    if len(generated_text.strip()) > 3:
        print("  ⚠ LLM输出了多余的文字，建议加强格式约束")
    elif not generated_text.strip().isdigit():
        print("  ⚠ LLM输出不是纯数字，解析可能不稳定")
    else:
        print("  ✓ LLM输出格式简洁，解析可靠")
    print("=" * 60)


if __name__ == "__main__":
    main()
