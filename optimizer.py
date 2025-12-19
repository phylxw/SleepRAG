import os
import json
import time
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai


# ================= 配置区域 =================

# 1. LLM 配置
MODEL_SOURCE = "huggingface"   # "huggingface" 或 "gemini"

HF_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 2. 文件路径
CLUSTERED_FILE = "AMATH-lighteval_auto_clustered_result.jsonl"
CLUSTER_SUMMARY_FILE = "AMATH-lighteval_cluster_summary.jsonl"
MEM_FREQ_FILE = "MATH-lighteval_memory_freq_20251218_150403.jsonl"
OUTPUT_OPTIMIZED_FILE = "AMATH-lighteval_optimized_memory_k50.jsonl"

# 3. 优化逻辑参数
TOP_K_HIGH = 50                # 高频 anchor 数量
BOTTOM_K_LOW = 50               # 低频扩写数量
LOW_FREQ_THRESHOLD = 2          # 频次阈值
TOP_N_SIMILAR_IN_CLUSTER = 5    # 类内合并邻居数

# 4. 相似度 embedding 模型
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# 5. LLM 并行与批量控制
# 对于多卡环境，增加 LLM_BATCH_SIZE 可以提高显卡利用率
LLM_BATCH_SIZE = 8          # 批量处理大小（根据显存调整，多卡可调大）
MAX_NEW_TOKENS = 512        # 输出长度
MAX_INPUT_TOKENS = 2048     # 输入长度
MAX_WORKERS = 4             # Gemini 并行请求数（仅在 MODEL_SOURCE="gemini" 时有效）

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
    """初始化 LLM"""
    global GLOBAL_MODEL, GLOBAL_TOKENIZER

    if MODEL_SOURCE == "gemini":
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            print(f"🤖 [Init] Gemini API ({GEMINI_MODEL_NAME}) 已配置")
        else:
            print("⚠️ [Init] 未检测到 GEMINI_API_KEY")
    elif MODEL_SOURCE == "huggingface":
        print(f"📥 [Init] 正在加载本地模型: {HF_MODEL_NAME} ...")
        try:
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
            # 必须设置 padding_side='left' 以支持批量推理
            GLOBAL_TOKENIZER.padding_side = 'left'
            if GLOBAL_TOKENIZER.pad_token is None:
                GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token
            
            # device_map="auto" 会自动将模型分布在多张显卡上
            GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_NAME,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            ).eval()
            print("✅ [Init] 本地模型多卡分发加载完成！")
        except Exception as e:
            print(f"❌ [Init] 本地模型加载失败: {e}")


def call_llm(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """单条调用接口"""
    # 包装批量接口
    res = call_llm_batch([prompt], max_new_tokens=max_new_tokens)
    return res[0] if res else ""


def call_llm_batch(prompts: List[str], max_new_tokens: int = MAX_NEW_TOKENS) -> List[str]:
    """批量推理接口：实现多卡并行/并发"""
    if not prompts:
        return []

    # --- Gemini：使用线程池模拟并行推理 ---
    if MODEL_SOURCE == "gemini":
        def single_gemini_call(p):
            try:
                model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                resp = model.generate_content(p)
                return clean_special_chars(resp.text.strip())
            except Exception:
                return ""
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            print(f" 🤖 [Gemini-Parallel] 正在并发处理 {len(prompts)} 条请求...")
            results = list(executor.map(single_gemini_call, prompts))
        return results

    # --- HuggingFace：利用 batching 和 device_map 进行硬件并行 ---
    if MODEL_SOURCE == "huggingface":
        if GLOBAL_MODEL is None:
            return [""] * len(prompts)

        try:
            print(f" 🚀 [Local-Batch] 正在并行生成 {len(prompts)} 条 (Batch Size={LLM_BATCH_SIZE})...", end="", flush=True)
            
            text_list = []
            for p in prompts:
                messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": p}]
                text = GLOBAL_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                text_list.append(text)

            model_inputs = GLOBAL_TOKENIZER(
                text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            ).to(GLOBAL_MODEL.device)

            with torch.no_grad():
                generated_ids = GLOBAL_MODEL.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=GLOBAL_TOKENIZER.pad_token_id
                )

            results = []
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
                new_token_ids = output_ids[len(input_ids):]
                text = GLOBAL_TOKENIZER.decode(new_token_ids, skip_special_tokens=True)
                results.append(clean_special_chars(text.strip()))
            print(" 完成")
            return results
        except Exception as e:
            print(f"\n❌ [Local Error]: {e}")
            return [""] * len(prompts)

    return [""] * len(prompts)


# =============== 任务 Prompt 构造 ===============

def get_summarize_prompt(group_texts: List[str]) -> str:
    """构造高频合并 Prompt"""
    items_formatted = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(group_texts))
    return f"""你是数学助教。下面是一组属于同一题型的记忆条目，它们都来自同一个聚类（同类问题）。
请将它们合并成**一条更完整、更抽象的记忆**，要求：
1. 不改变任何结论，也不要引入新的数值或额外事实。
2. 保留所有关键条件、公式与解题结论。
3. 适当总结共同的解题思路，可以合并重复信息。
4. 用English写成一段或两段连续文本，不要分条列出原题号。

待合并的记忆条目如下：
{items_formatted}
"""

def get_expand_prompt(text: str) -> str:
    """构造低频扩写 Prompt"""
    return f"""你是数学助教。下面是一条数学题目的记忆内容。
请在 **不改变题目条件和答案、不添加任何新数值或事实** 的前提下，对它进行语义扩写：
1. 可以增加对题目考察点的解释和背景说明。
2. 可以加入同义改写、更多自然语言表述，以便未来更容易被检索到。
3. 输出一段或两段English文本，不要丢失原始信息。

原始记忆：
{text}
"""


# =============== 数据加载与向量计算 (保持原逻辑) ===============

def load_clustered_memories(path: str) -> Tuple[Dict[str, dict], List[str]]:
    memories, order = {}, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            mid = str(obj["id"])
            memories[mid] = obj
            order.append(mid)
    return memories, order

def load_cluster_summary(path: str) -> Dict[int, List[str]]:
    cluster_to_ids = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            cluster_to_ids[int(obj["cluster_id"])] = [str(x) for x in obj.get("memory_ids", [])]
    return cluster_to_ids

def load_memory_freq(path: str) -> Dict[str, int]:
    freq_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            mid = str(obj.get("memory_id", obj.get("id", "")))
            if mid: freq_map[mid] = int(obj.get("freq", 0))
    return freq_map

def build_embeddings_for_memories(memories: Dict[str, dict]) -> Dict[str, np.ndarray]:
    device = "cuda" if has_cuda() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    ids = list(memories.keys())
    texts = [memories[mid].get("question") or memories[mid].get("contents", "") for mid in ids]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return {mid: embeddings[i] for i, mid in enumerate(ids)}

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def select_high_low_ids(freq_map: Dict[str, int], top_k_high: int, bottom_k_low: int, low_freq: int):
    items = sorted(freq_map.items(), key=lambda x: -x[1])
    high_ids = [mid for mid, f in items[:top_k_high]]
    items_asc = sorted(freq_map.items(), key=lambda x: x[1])
    low_ids, zero_ids = [], []
    for mid, f in items_asc:
        if f == 0: zero_ids.append(mid)
        elif f == low_freq and len(low_ids) < bottom_k_low: low_ids.append(mid)
    return set(high_ids), set(low_ids), set(zero_ids)


# =============== 主优化逻辑 (重点修改：批量并行) ===============

def optimize_memory():
    init_llm()
    memories, id_order = load_clustered_memories(CLUSTERED_FILE)
    cluster_to_ids = load_cluster_summary(CLUSTER_SUMMARY_FILE)
    freq_map = load_memory_freq(MEM_FREQ_FILE)
    for mid in memories: freq_map.setdefault(mid, 0)

    high_ids, low_ids, zero_ids = select_high_low_ids(freq_map, TOP_K_HIGH, BOTTOM_K_LOW, LOW_FREQ_THRESHOLD)
    id_to_emb = build_embeddings_for_memories(memories)

    to_delete_ids = set(zero_ids)
    merged_consumed_ids = set()

    # --- 阶段 4：高频聚合 (批量收集模式) ---
    print("\n========== 高频记忆聚合阶段 (多卡并行准备) ==========")
    high_ids_sorted = sorted(list(high_ids), key=lambda x: -freq_map.get(x, 0))
    
    aggregation_tasks = [] # 存储 (anchor_id, neighbors, prompt)

    for anchor_id in high_ids_sorted:
        if anchor_id not in memories or anchor_id in merged_consumed_ids: continue
        
        rec_anchor = memories[anchor_id]
        cid = rec_anchor.get("cluster_id")
        if cid is None: continue
        
        members = [str(x) for x in cluster_to_ids.get(int(cid), [])]
        candidates = [m for m in members if m != anchor_id and m not in merged_consumed_ids]
        if not candidates: continue

        anchor_emb = id_to_emb.get(anchor_id)
        sims = [(m, cosine_similarity(anchor_emb, id_to_emb[m])) for m in candidates if m in id_to_emb]
        if not sims: continue

        neighbors = [m for m, _ in sorted(sims, key=lambda x: -x[1])[:TOP_N_SIMILAR_IN_CLUSTER]]
        
        # 预备文本
        group_ids = [anchor_id] + neighbors
        group_texts = [f"[ID {mid}] {memories[mid].get('question') or memories[mid].get('contents', '')}" for mid in group_ids]
        
        # 记录任务
        aggregation_tasks.append({
            "anchor_id": anchor_id,
            "neighbors": neighbors,
            "prompt": get_summarize_prompt(group_texts)
        })

        # 标记消耗
        for mid in neighbors: merged_consumed_ids.add(mid)

    # 批量执行高频聚合
    if aggregation_tasks:
        prompts = [t["prompt"] for t in aggregation_tasks]
        results = []
        for i in range(0, len(prompts), LLM_BATCH_SIZE):
            batch = prompts[i : i + LLM_BATCH_SIZE]
            results.extend(call_llm_batch(batch))

        for task, summary in zip(aggregation_tasks, results):
            if not summary: continue
            aid = task["anchor_id"]
            neighbors = task["neighbors"]
            
            rec = memories[aid]
            rec["original_question"] = rec.get("question") or rec.get("contents", "")
            rec["question"] = summary
            rec["merged_from_ids"] = [aid] + neighbors
            rec["merge_type"] = "high_freq_anchor"
            
            for mid in neighbors:
                if freq_map.get(mid, 0) < LOW_FREQ_THRESHOLD:
                    to_delete_ids.add(mid)

    # --- 阶段 5：低频扩写 (批量收集模式) ---
    print("\n========== 低频记忆扩写阶段 (多卡并行准备) ==========")
    low_expand_ids = [mid for mid in low_ids if mid in memories and mid not in to_delete_ids]
    
    if low_expand_ids:
        expand_prompts = [get_expand_prompt(memories[mid].get("question") or memories[mid].get("contents", "")) for mid in low_expand_ids]
        expand_results = []
        for i in range(0, len(expand_prompts), LLM_BATCH_SIZE):
            batch = expand_prompts[i : i + LLM_BATCH_SIZE]
            expand_results.extend(call_llm_batch(batch))

        for mid, expanded in zip(low_expand_ids, expand_results):
            if not expanded: continue
            rec = memories[mid]
            rec["original_question"] = rec.get("question") or rec.get("contents", "")
            rec["question"] = expanded
            rec["opt_type"] = "low_freq_expanded"

    # --- 阶段 6：写出结果 ---
    print("\n========== 写出优化后的记忆库 ==========")
    kept_count = 0
    with open(OUTPUT_OPTIMIZED_FILE, "w", encoding="utf-8") as f:
        for mid in id_order:
            if mid in memories and mid not in to_delete_ids:
                f.write(json.dumps(memories[mid], ensure_ascii=False) + "\n")
                kept_count += 1

    print(f"✅ 完成！保留: {kept_count}, 删除: {len(to_delete_ids)}")


if __name__ == "__main__":
    optimize_memory()