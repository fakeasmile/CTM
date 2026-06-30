"""生成形容词概念向量（Chat Template + vLLM）—— 直接评估版

核心思路：让LLM直接评估文本与各形容词之间的相关程度，
通过解析LLM的文本输出提取1~5评分，构建可解释的概念向量。

【直接评估流程】
1. 构建Chat Template prompt（1条文本 + 1个形容词 + 定义）→ LLM推理（max_tokens=32）
2. LLM输出评分（如"5"或"评分：5"）
3. 解析输出文本中的1~5整数评分
4. 归一化到[0, 1]：score = (rating - 1) / 4
5. 每条文本遍历所有形容词，得到177维概念向量

使用示例：
python scripts/generate_adjective_c_r_vllm.py --mode train --model_name glm-4-9b-chat
python scripts/generate_adjective_c_r_vllm.py --mode test --model_name Qwen2.5-7B-Instruct
python scripts/generate_adjective_c_r_vllm.py --mode test --model_name glm-4-9b-chat --adjective_name toxic_adjectives_v3.csv
"""

import argparse
import os
import re
import sys
from pathlib import Path
import json

# AutoDL环境中OMP_NUM_THREADS可能被设为无效值，导致vLLM报错，需清理
if "OMP_NUM_THREADS" in os.environ:
    val = os.environ["OMP_NUM_THREADS"].strip()
    if not val.isdigit() or int(val) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# =============================================================================
# 项目路径
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ADJ_DIR = DATA_DIR / "adjective"
TOXICN_DIR = DATA_DIR / "TOXICN"
OUTPUT_DIR = PROJECT_ROOT / "output" / "experiments"
MODELS_PATH = PROJECT_ROOT / "models"  # 本地模型存放目录


# =============================================================================
# 命令行参数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="生成形容词概念向量（vLLM直接评估版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help='train:生成训练集的概念向量，test:生成测试集的概念向量')
    parser.add_argument('--model_name', type=str, required=True, help='LLM模型名称（也是models/下的子目录名）')
    parser.add_argument('--adjective_name', type=str, default='toxic_adjectives_v1.csv',
                        help='形容词词典文件名，默认toxic_adjectives_v1.csv')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                        help='vLLM GPU显存占用比例（0.0-1.0），默认0.85')
    parser.add_argument('--threshold', type=float, default=1e-4,
                        help='概念向量截断阈值，小于该值的分数设为0')
    return parser.parse_args()


# =============================================================================
# 模型加载配置表
# =============================================================================
# 所有模型相关的加载参数均集中在此配置表中，保证LLM切换对后续流程透明。
# 新增模型只需在本字典中增加条目，通常无需修改核心推理逻辑。
# prompt_suffix：部分模型在chat template后需要追加后缀
#   - GLM-4：首token为\n，追加\n使其直接输出数字
#   - Qwen：首token带空格，已在提示词末尾加空格处理，suffix为空
MODEL_LOADING_CONFIG = {
    "Qwen2.5-7B-Instruct": {
        "quantization": None,
        "is_qwen3": False,
        "is_multimodal": False,
        "prompt_suffix": "",
    },
    "Qwen3.5-9B": {
        "quantization": "fp8",       # FP8在线量化，加速推理
        "is_qwen3": True,            # 需禁用thinking模式
        "is_multimodal": True,       # 需跳过视觉编码器
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
    """从 MODEL_LOADING_CONFIG 中获取模型加载配置，未知模型直接报错。"""
    if model_name not in MODEL_LOADING_CONFIG:
        raise ValueError(
            f"不支持的模型: {model_name}。请在 MODEL_LOADING_CONFIG 中添加该模型的配置条目后重试。"
        )
    return MODEL_LOADING_CONFIG[model_name].copy()


# =============================================================================
# 模型加载
# =============================================================================
def load_vllm_model(model_path: Path, model_name: str, gpu_memory_utilization: float = 0.85):
    """加载vLLM模型和tokenizer。

    模型差异（量化、多模态、Qwen3+）均从MODEL_LOADING_CONFIG读取。
    Returns: (tokenizer, llm, is_qwen3)
    """
    llm_path = model_path / model_name
    if not llm_path.exists():
        raise ValueError(f"LLM path {llm_path} does not exist")

    model_config = get_model_loading_config(model_name)
    quantization = model_config["quantization"]
    is_multimodal = model_config["is_multimodal"]

    # 加载tokenizer
    print(f"Loading tokenizer from {llm_path}")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构建vLLM加载参数
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


# =============================================================================
# 提示词定义
# =============================================================================
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


# =============================================================================
# Prompt构建
# =============================================================================
def build_chat_messages(content, adj, adj_definition=None):
    """构建直接评估的Chat Template messages。

    user_content结构：
        文本内容：{content}
        形容词：{adj}
        定义：{adj_definition}  ← 仅当定义存在时插入
        该文本在多大程度上体现了"{adj}"所描述的特征？
        请评分（1-5）：
    """
    user_lines = [f"文本内容：{content}"]
    user_lines.append(f"形容词：{adj}")
    if adj_definition:
        user_lines.append(f"定义：{adj_definition}")
    user_lines.append(f"该文本在多大程度上体现了\"{adj}\"所描述的特征？")
    user_lines.append("请评分（1-5）：")
    user_content = "\n".join(user_lines)

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


# =============================================================================
# 评分解析
# =============================================================================
def parse_rating(output_text: str) -> int:
    """从LLM输出文本中解析1~5的评分。

    解析策略（按优先级）：
    1. 提取"评分"关键字后的数字
    2. 提取首个独立数字（1-5）
    3. 中文数字映射
    4. 找不到则返回1（保守默认值）

    Returns:
        int: 1~5的评分
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
    """将1~5评分归一化到[0, 1]。

    映射：1→0.0, 2→0.25, 3→0.5, 4→0.75, 5→1.0
    """
    return (rating - 1) / 4.0


# =============================================================================
# 核心流程：生成形容词概念向量
# =============================================================================
def generate_adj_concept(data_path, output_path, csv_output_path, adjective_path,
                         tokenizer, llm_model,
                         is_qwen3=False, prompt_suffix="", threshold=1e-4):
    """生成形容词概念向量。

    对数据集中每条文本，遍历所有形容词，通过LLM直接评估相关程度，
    构建概念向量（每条文本一个V维向量，V=形容词数量）。

    流程：
    1. 加载形容词词典和数据集
    2. 逐文本处理：构建prompt → vLLM推理 → 解析评分 → 归一化
    3. 保存结果（JSON含完整信息，CSV为纯矩阵）
    """
    # --- 准备工作 ---
    # 加载形容词词典
    adj_df = pd.read_csv(adjective_path)
    adjectives = adj_df["chinese"].tolist()
    adj_definitions = adj_df["definition"].tolist() if "definition" in adj_df.columns else [None] * len(adjectives)
    num_adjs = len(adjectives)

    # 加载数据集
    with open(data_path, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    print(f"形容词数: {num_adjs}")
    print(f"样本数: {len(data_set)}")

    # --- LLM直接评估 ---
    sampling_params = SamplingParams(max_tokens=32, temperature=0)

    results = []
    concept_matrix = []  # [N, V] 矩阵，用于CSV输出
    parse_fail_count = 0

    for sample_idx, sample in enumerate(tqdm(data_set, desc="Processing samples")):
        content = sample["content"]

        # 为当前文本构建所有形容词的prompt
        prompts = []
        for adj, adj_def in zip(adjectives, adj_definitions):
            messages = build_chat_messages(content, adj, adj_def)
            chat_template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            )
            prompt_text += prompt_suffix
            prompts.append(prompt_text)

        # 批量推理：一次性送入当前文本的所有prompt（vLLM自动调度）
        outputs = llm_model.generate(prompts, sampling_params, use_tqdm=False)

        # 从每条推理结果中解析评分
        concept_vector = []
        raw_ratings = []
        for adj_idx, sample_info in enumerate(outputs):
            generated_text = sample_info.outputs[0].text.strip()
            rating = parse_rating(generated_text)
            score = rating_to_score(rating)
            concept_vector.append(score)
            raw_ratings.append(rating)

            # 统计解析失败（默认1分的）
            if rating == 1 and "1" not in generated_text[:5]:
                parse_fail_count += 1

        # 防御性校验
        if len(concept_vector) != num_adjs:
            raise RuntimeError(f"concept_vector长度异常：期望{num_adjs}，实际{len(concept_vector)}")

        # 截断极小值（低于阈值的分数设为0，避免浮点噪声）
        truncated_vector = [s if abs(s) >= threshold else 0.0 for s in concept_vector]
        concept_matrix.append(truncated_vector)

        # 组装结果
        result_item = {
            "content": sample["content"],
            "toxic": sample["toxic"],
            "concept": truncated_vector,
            "ratings": raw_ratings,
        }
        results.append(result_item)

    # --- 保存结果 ---
    # JSON：含完整信息（content, toxic, concept, ratings）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"形容词概念向量(JSON)保存到: {output_path}")

    # CSV：纯矩阵 [N, V]，横轴为文本，纵轴为形容词
    df = pd.DataFrame(concept_matrix, columns=adjectives)
    df.insert(0, "content", [r["content"] for r in results])
    df.insert(1, "toxic", [r["toxic"] for r in results])
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"形容词概念向量(CSV)保存到: {csv_output_path}")
    print(f"矩阵形状: [{len(concept_matrix)}, {len(adjectives)}] (文本数, 形容词数)")
    print(f"截断阈值: {threshold}，小于该值的分数已设为0")

    # 评分分布统计
    all_ratings = [r for res in results for r in res["ratings"]]
    rating_counts = {i: all_ratings.count(i) for i in range(1, 6)}
    total_ratings = len(all_ratings)
    print(f"\n评分分布:")
    for r, c in rating_counts.items():
        print(f"  {r}分: {c:,} ({100*c/total_ratings:.1f}%)")
    if parse_fail_count > 0:
        print(f"  解析失败(默认1分): {parse_fail_count:,}")


# =============================================================================
# 主入口
# =============================================================================
def main():
    args = parse_args()

    # 构建路径
    data_path = TOXICN_DIR / f"{args.mode}.json"
    adjective_path = ADJ_DIR / args.adjective_name

    if not data_path.exists():
        raise FileNotFoundError(f"数据集不存在: {data_path}")
    if not adjective_path.exists():
        raise FileNotFoundError(f"形容词词典不存在: {adjective_path}")

    # 从词典文件名提取词干用于输出文件命名（如 toxic_adjectives_v1.csv → v1）
    adj_stem = adjective_path.stem  # toxic_adjectives_v1
    adj_version = adj_stem.replace("toxic_adjectives_", "")  # v1

    # 输出目录
    concept_dir = OUTPUT_DIR / args.model_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{adj_version}.json"
    csv_output_path = concept_dir / f"concept_{args.mode}_{args.model_name}_{adj_version}.csv"

    # 打印配置
    print("\n" + "=" * 60)
    print("形容词概念向量生成(vLLM直接评估版) - 配置信息")
    print("=" * 60)
    print(f"LLM模型名称: {args.model_name}")
    print(f"形容词词典: {adjective_path.name} ({adjective_path})")
    print(f"当前模式: {args.mode}")
    print(f"GPU显存占用比例: {args.gpu_memory_utilization}")
    print(f"数据集路径: {data_path}")
    print(f"JSON输出路径: {output_path}")
    print(f"CSV输出路径: {csv_output_path}")
    print("=" * 60 + "\n")

    # 加载模型
    tokenizer, llm_model, qwen3_flag = load_vllm_model(
        MODELS_PATH, args.model_name, args.gpu_memory_utilization
    )
    if qwen3_flag:
        print(f"检测到Qwen3+模型({args.model_name})，已禁用思考模式(enable_thinking=False)")

    # 获取模型特定配置
    model_config = get_model_loading_config(args.model_name)
    prompt_suffix = model_config.get("prompt_suffix", "")
    if prompt_suffix:
        print(f"检测到模型({args.model_name})需要追加prompt后缀: {repr(prompt_suffix)}")

    # 执行概念向量生成
    generate_adj_concept(
        data_path, output_path, csv_output_path, adjective_path,
        tokenizer, llm_model,
        is_qwen3=qwen3_flag, prompt_suffix=prompt_suffix, threshold=args.threshold,
    )

    print("生成完成")


if __name__ == '__main__':
    main()
