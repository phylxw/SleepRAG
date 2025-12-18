import os
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai


# ================= 配置区域 =================

# 1. LLM 配置：和你聚类文件保持一致
MODEL_SOURCE = "huggingface"   # "huggingface" 或 "gemini"

HF_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 2. 文件路径（你可以根据数据集改名；这里先以 MATH 为例）
CLUSTERED_FILE = "math_auto_clustered_result.jsonl"           # 聚类后的记忆文件
CLUSTER_SUMMARY_FILE = "math_cluster_summary.jsonl"           # 每个类有哪些记忆ID
MEM_FREQ_FILE = "MATH-lighteval_memory_freq_20251216_122715.jsonl"  # 调用频次文件
OUTPUT_OPTIMIZED_FILE = "MATH_optimized_memory_k200.jsonl"   # 输出的新记忆库

# 3. 优化逻辑参数
TOP_K_HIGH = 30                # 作为“高频记忆 anchor”的条目数量（按频次排序）
BOTTOM_K_LOW = 30              # 作为“低频记忆扩写对象”的条目数量（按频次从低到高）
LOW_FREQ_THRESHOLD = 2          # 被高频合并时，如果 freq < 这个阈值就直接删掉
TOP_N_SIMILAR_IN_CLUSTER = 5    # 高频 anchor 在类内选 top-n 相似记忆来合并

# 4. 相似度 embedding 模型
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# ===========================================

GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None


# =============== 工具函数 ===============

def clean_special_chars(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.replace('\u2028', ' ').replace('\u2029', ' ')


def has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# =============== LLM 初始化与调用 ===============

def init_llm():
    """初始化 LLM（和你的聚类脚本保持一致风格）"""
    global GLOBAL_MODEL, GLOBAL_TOKENIZER

    if MODEL_SOURCE == "gemini":
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            print(f"🤖 [Init] Gemini API ({GEMINI_MODEL_NAME}) 已配置")
        else:
            print("⚠️ [Init] 未检测到 GEMINI_API_KEY，Gemini 相关功能会被跳过")
    elif MODEL_SOURCE == "huggingface":
        print(f"📥 [Init] 正在加载本地模型: {HF_MODEL_NAME} ...")
        try:
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
            GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_NAME,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            ).eval()
            print("✅ [Init] 本地模型加载完成！")
        except Exception as e:
            print(f"❌ [Init] 本地模型加载失败: {e}")
            print("💡 提示: 请检查 HuggingFace 权限和网络")


def call_llm(prompt: str, max_new_tokens: int = 256) -> str:
    """统一的大模型调用接口（Gemini / 本地 Qwen）"""

    # --- Gemini ---
    if MODEL_SOURCE == "gemini":
        if not GEMINI_API_KEY:
            return "Skipped (No GEMINI_API_KEY)"
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            print("  🤖 [Gemini] 正在生成...", end="", flush=True)
            resp = model.generate_content(prompt)
            print(" 完成")
            return clean_special_chars(resp.text.strip())
        except Exception as e:
            print(f"\n❌ [Gemini Error]: {e}")
            return ""

    # --- HuggingFace 本地 ---
    elif MODEL_SOURCE == "huggingface":
        if GLOBAL_MODEL is None:
            print("⚠️ [Local] LLM 尚未初始化")
            return ""

        try:
            print("  🚀 [Local] 正在生成...", end="", flush=True)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = GLOBAL_TOKENIZER.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = GLOBAL_TOKENIZER([text], return_tensors="pt").to(GLOBAL_MODEL.device)
            with torch.no_grad():
                generated_ids = GLOBAL_MODEL.generate(
                    model_inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            # 只取新增的部分
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = GLOBAL_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(" 完成")
            return clean_special_chars(response.strip())
        except Exception as e:
            print(f"\n❌ [Local Error]: {e}")
            return ""

    return ""


# ===== 高频 & 低频记忆的 LLM 操作 =====

def summarize_high_freq_memory(anchor_id: str, group_texts: List[str]) -> str:
    """
    高频记忆类内聚合：给定 anchor + 同类若干条相似记忆，把它们合并成一个更“深”的记忆。
    """
    items_formatted = "\n".join(
        f"[{i+1}] {t}" for i, t in enumerate(group_texts)
    )
    prompt = f"""你是数学助教。下面是一组属于同一题型的记忆条目，它们都来自同一个聚类（同类问题）。
请将它们合并成**一条更完整、更抽象的记忆**，要求：

1. 不改变任何结论，也不要引入新的数值或额外事实。
2. 保留所有关键条件、公式与解题结论。
3. 适当总结共同的解题思路，可以合并重复信息。
4. 用English写成一段或两段连续文本，不要分条列出原题号。

待合并的记忆条目如下：
{items_formatted}
"""
    return call_llm(prompt)


def expand_low_freq_memory(text: str) -> str:
    """
    低频记忆扩写：不改变核心语义、不新增事实，只做解释 & 同义扩写。
    """
    prompt = f"""你是数学助教。下面是一条数学题目的记忆内容。

请在 **不改变题目条件和答案、不添加任何新数值或事实** 的前提下，对它进行语义扩写：
1. 可以增加对题目考察点的解释和背景说明。
2. 可以加入同义改写、更多自然语言表述，以便未来更容易被检索到。
3. 输出一段或两段English文本，不要丢失原始信息。

原始记忆：
{text}
"""
    return call_llm(prompt)


# =============== 数据加载 ===============

def load_clustered_memories(path: str) -> Tuple[Dict[str, dict], List[str]]:
    """
    读取 *_auto_clustered_result.jsonl
    返回：
      - id -> 记录 dict
      - id_list: 保留原始顺序的 id 列表（方便最后写回）
    """
    memories: Dict[str, dict] = {}
    order: List[str] = []
    print(f"📥 正在加载聚类后的记忆文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            mid = str(obj["id"])
            memories[mid] = obj
            order.append(mid)
    print(f"✅ 共加载 {len(memories)} 条记忆")
    return memories, order


def load_cluster_summary(path: str) -> Dict[int, List[str]]:
    """
    读取 *_cluster_summary.jsonl
    返回：cluster_id -> [memory_ids...]
    """
    cluster_to_ids: Dict[int, List[str]] = {}
    print(f"📥 正在加载聚类摘要文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = int(obj["cluster_id"])
            ids = [str(x) for x in obj.get("memory_ids", [])]
            cluster_to_ids[cid] = ids
    print(f"✅ 共加载 {len(cluster_to_ids)} 个聚类")
    return cluster_to_ids


def load_memory_freq(path: str) -> Dict[str, int]:
    """
    读取调用频次文件 MATH-lighteval_memory_freq_*.jsonl
    预期每行包含 memory_id / id, freq 字段。
    """
    freq_map: Dict[str, int] = {}
    print(f"📥 正在加载记忆频次文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            mid = str(obj.get("memory_id", obj.get("id", "")))
            if not mid:
                continue
            freq = int(obj.get("freq", 0))
            freq_map[mid] = freq
    print(f"✅ 频次记录数: {len(freq_map)}")
    return freq_map


# =============== Embedding & 相似度 ===============

def build_embeddings_for_memories(memories: Dict[str, dict]) -> Dict[str, np.ndarray]:
    """
    对所有记忆构建向量，用于类内相似度计算。
    默认为使用记录中的 "question" 字段；如果你想改成 "contents" 就自己换一下。
    """
    device = "cuda" if has_cuda() else "cpu"
    print(f"🚀 正在计算记忆向量 ({EMBEDDING_MODEL}) on {device}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    ids = list(memories.keys())
    texts = []
    for mid in ids:
        rec = memories[mid]
        text = rec.get("question") or rec.get("contents", "")
        texts.append(text)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    id_to_emb = {mid: embeddings[i] for i, mid in enumerate(ids)}
    print(f"✅ 向量构建完成，共 {len(id_to_emb)} 条")
    return id_to_emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# =============== 高频/低频集合选择 ===============

def select_high_low_ids(
    freq_map: Dict[str, int],
    top_k_high: int,
    bottom_k_low: int,
    low_freq_for_low_only: int = 1
):
    """
    从 freq_map 中选：
      - top_k_high 个最高频作为高频 anchor
      - bottom_k_low 个低频候选（但只保留 freq == low_freq_for_low_only 的）
      - 同时记录所有 freq == 0 的 id 方便之后删除
    """
    # 先补全 0 频次（如果有一些 id 在 freq_map 中没出现）
    # （这一补全应该在外部对所有 IDs 做一次，这里假设已经补）
    items = list(freq_map.items())
    # 高频：按 freq 降序
    sorted_desc = sorted(items, key=lambda x: -x[1])
    high_ids = [mid for mid, f in sorted_desc[:top_k_high]]

    # 低频：按 freq 升序
    sorted_asc = sorted(items, key=lambda x: x[1])
    low_ids = []
    zero_ids = []
    for mid, f in sorted_asc:
        if f == 0:
            zero_ids.append(mid)
            continue
        if f == low_freq_for_low_only:
            low_ids.append(mid)
        if len(low_ids) >= bottom_k_low:
            break

    print(f"🔥 高频 anchor 数量: {len(high_ids)}")
    print(f"🧊 0 次调用的记忆数量: {len(zero_ids)}（之后会删除）")
    print(f"🥶 低频扩写候选(freq={low_freq_for_low_only})数量: {len(low_ids)} (最多 bottom_k={bottom_k_low})")
    return set(high_ids), set(low_ids), set(zero_ids)


# =============== 主优化逻辑 ===============

def optimize_memory():
    # 0. 初始化 LLM
    init_llm()

    # 1. 读入基础数据
    memories, id_order = load_clustered_memories(CLUSTERED_FILE)
    cluster_to_ids = load_cluster_summary(CLUSTER_SUMMARY_FILE)
    freq_map = load_memory_freq(MEM_FREQ_FILE)

    # 为所有记忆补齐频次
    for mid in memories.keys():
        freq_map.setdefault(mid, 0)

    # 2. 选出高频、低频、0 频集合
    high_ids, low_ids, zero_ids = select_high_low_ids(
        freq_map,
        TOP_K_HIGH,
        BOTTOM_K_LOW,
        low_freq_for_low_only=LOW_FREQ_THRESHOLD
    )

    # 3. 准备向量，用于类内相似度
    id_to_emb = build_embeddings_for_memories(memories)

    # 4. 高频：类内聚合（merge）
    merged_consumed_ids = set()      # 被当作“邻居”参与 merge 的记忆 id
    to_delete_ids = set()            # 最终要彻底删除的 id（低频被 merge / 频次为0 等）

    print("\n========== 高频记忆聚合阶段 ==========")
    # 按频次从高到低顺序处理 anchor，避免 rank 低的 anchor 抢走高频邻居
    high_ids_sorted = sorted(list(high_ids), key=lambda x: -freq_map.get(x, 0))

    for anchor_id in high_ids_sorted:
        if anchor_id not in memories:
            continue
        if anchor_id in merged_consumed_ids:
            # 说明已经作为别人 group 的成员了，就不再当 anchor
            continue

        rec_anchor = memories[anchor_id]
        cluster_id = rec_anchor.get("cluster_id")
        if cluster_id is None:
            continue

        cluster_id = int(cluster_id)
        cluster_member_ids = [str(x) for x in cluster_to_ids.get(cluster_id, [])]
        if not cluster_member_ids:
            continue

        # 候选邻居：同类、不是自己、没被 merge 过
        candidates = [
            mid for mid in cluster_member_ids
            if mid != anchor_id and mid not in merged_consumed_ids
        ]
        if not candidates:
            continue

        anchor_emb = id_to_emb.get(anchor_id)
        if anchor_emb is None:
            continue

        sims = []
        for mid in candidates:
            emb = id_to_emb.get(mid)
            if emb is None:
                continue
            sims.append((mid, cosine_similarity(anchor_emb, emb)))

        if not sims:
            continue

        # 取类内 top-n 相似
        sims_sorted = sorted(sims, key=lambda x: -x[1])
        neighbors = [mid for mid, _ in sims_sorted[:TOP_N_SIMILAR_IN_CLUSTER]]
        group_ids = [anchor_id] + neighbors

        print(f"\n🔥 Anchor {anchor_id} (freq={freq_map[anchor_id]}, cluster={cluster_id})")
        print(f"   合并同类 top-{len(neighbors)}: {neighbors}")

        # 构造要给 LLM 的文本
        group_texts = []
        for mid in group_ids:
            rec = memories[mid]
            text = rec.get("question") or rec.get("contents", "")
            group_texts.append(f"[ID {mid}] {text}")

        summary_text = summarize_high_freq_memory(anchor_id, group_texts)
        if not summary_text:
            print("   ⚠️ LLM 返回为空，跳过这组合并")
            continue

        # 更新 anchor 的内容：用 summary 替换 question，并保留原始信息
        original_text = rec_anchor.get("question") or rec_anchor.get("contents", "")
        rec_anchor["original_question"] = original_text
        rec_anchor["question"] = summary_text
        rec_anchor["merged_from_ids"] = group_ids
        rec_anchor["merge_type"] = "high_freq_anchor"

        # 邻居标记为已参与 merge；其中低频的标记为删除
        for mid in neighbors:
            merged_consumed_ids.add(mid)
            if freq_map.get(mid, 0) < LOW_FREQ_THRESHOLD:
                to_delete_ids.add(mid)

    # 5. 低频：不被合并消掉、freq=1 的记忆做扩写
    print("\n========== 低频记忆扩写阶段 ==========")
    # 先把所有 freq=0 的直接加入删除集合
    to_delete_ids.update(zero_ids)

    # 真正要扩写的低频记忆：freq==1，且没有被 merge 消耗掉
    low_expand_ids = [
        mid for mid in low_ids
        if mid in memories and mid not in to_delete_ids
    ]

    print(f"🥶 需要扩写的低频记忆条目数: {len(low_expand_ids)}")

    for mid in low_expand_ids:
        rec = memories[mid]
        base_text = rec.get("question") or rec.get("contents", "")
        print(f"\n🥶 扩写低频记忆 ID={mid}, freq={freq_map[mid]}")
        expanded = expand_low_freq_memory(base_text)
        if not expanded:
            print("   ⚠️ LLM 返回为空，保持原文不变")
            continue

        rec["original_question"] = base_text
        rec["question"] = expanded
        rec["opt_type"] = "low_freq_expanded"

    # 6. 写出新的记忆库：跳过 to_delete_ids
    print("\n========== 写出优化后的记忆库 ==========")
    kept_count = 0
    with open(OUTPUT_OPTIMIZED_FILE, "w", encoding="utf-8") as f:
        for mid in id_order:
            if mid not in memories:
                continue
            if mid in to_delete_ids:
                continue
            f.write(json.dumps(memories[mid], ensure_ascii=False) + "\n")
            kept_count += 1

    print(f"✅ 新记忆库写入完成: {OUTPUT_OPTIMIZED_FILE}")
    print(f"   保留记忆条目: {kept_count}")
    print(f"   删除记忆条目: {len(to_delete_ids)}")
    print("   （注意：原始 *_auto_clustered_result.jsonl 文件没有被修改）")


if __name__ == "__main__":
    optimize_memory()