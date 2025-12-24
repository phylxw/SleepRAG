import os
import json
import time
from typing import Dict, List, Tuple, Set
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
# Hydra
import hydra
from omegaconf import DictConfig
from utils.toolfunction import clean_special_chars,has_cuda
from tools.optimize.callllm import init_llm,call_llm,call_llm_batch
from tools.optimize.memoryload import load_clustered_memories,load_cluster_summary,load_memory_freq
# ================= 全局变量 (保持原逻辑) =================
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_SGLANG_CLIENT = None

# ===== 高频 & 低频记忆的 LLM 操作 =====
def summarize_high_freq_prompt(group_texts: List[str], cfg: DictConfig) -> str:
    items_formatted = "\n".join(
        f"[{i+1}] {t}" for i, t in enumerate(group_texts)
    )
    template = cfg.optimizer.prompts.summarize_high_freq
    prompt = template.format(items_formatted=items_formatted)
    return prompt

def expand_low_freq_memory_prompt(text: str, cfg: DictConfig) -> str:
    """构造低频记忆扩写的 prompt"""
    template = cfg.optimizer.prompts.expand_low_freq
    prompt = template.format(text=text)
    
    return prompt


# =============== Embedding & 相似度 ===============

def build_embeddings_for_memories(memories: Dict[str, dict], model_name: str) -> Dict[str, np.ndarray]:
    device = "cuda" if has_cuda() else "cpu"
    print(f"🚀 正在计算记忆向量 ({model_name}) on {device}...")
    model = SentenceTransformer(model_name, device=device)

    ids = list(memories.keys())
    texts = []
    for mid in ids:
        rec = memories[mid]
        text = rec.get("contents", "")
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
    items = list(freq_map.items())
    # 高频：按 freq 降序
    sorted_desc = sorted(items, key=lambda x: -x[1])
    high_ids = [mid for mid, f in sorted_desc[:top_k_high]]

    # 低频：按 freq 升序
    sorted_asc = sorted(items, key=lambda x: x[1])
    low_ids = []
    zero_ids = []
    for mid, f in sorted_asc:
        if f < 0:
            zero_ids.append(mid)
            continue
        if f == low_freq_for_low_only:
            low_ids.append(mid)
        if len(low_ids) >= bottom_k_low:
            break

    print(f"🔥 高频 anchor 数量: {len(high_ids)}")
    print(f"🧊 分数小于-2的记忆数量: {len(zero_ids)}（之后会删除）")
    print(f"🥶 低频扩写候选(freq={low_freq_for_low_only})数量: {len(low_ids)} (最多 bottom_k={bottom_k_low})")
    return set(high_ids), set(low_ids), set(zero_ids)


# =============== 主优化逻辑 (Hydra Managed) ===============

@hydra.main(version_base=None, config_path="conf", config_name="config")
def optimize_memory(cfg: DictConfig):
    # 0. 初始化 LLM
    init_llm(cfg)

    # 1. 读入基础数据 (使用 config 中的路径)
    cluster_file = cfg.paths.cluster_output
    summary_file = cfg.paths.cluster_summary
    freq_file = cfg.paths.freq_file
    output_file = cfg.paths.optimized_memory

    memories, id_order = load_clustered_memories(cluster_file)
    cluster_to_ids = load_cluster_summary(summary_file)
    freq_map = load_memory_freq(freq_file)

    if not memories:
        print("❌ 无法加载记忆数据，程序退出。")
        return

    # 为所有记忆补齐频次
    for mid in memories.keys():
        freq_map.setdefault(mid, 0)

    # 2. 选出高频、低频、0 频集合 (使用 config 中的参数)
    high_ids, low_ids, zero_ids = select_high_low_ids(
        freq_map,
        top_k_high=cfg.optimizer.top_k_high,
        bottom_k_low=cfg.optimizer.bottom_k_low,
        low_freq_for_low_only=cfg.optimizer.low_freq_threshold
    )

    # 3. 准备向量
    id_to_emb = build_embeddings_for_memories(memories, cfg.model.embedding_name)

# 4. 高频：类内聚合（merge）
    merged_consumed_ids = set()      
    to_delete_ids = set()            

    print("\n========== 高频记忆聚合阶段 (Batch Optimized) ==========")
    to_delete_ids.update(zero_ids)
    high_ids_sorted = sorted(list(high_ids), key=lambda x: -freq_map.get(x, 0))
    top_n_similar = cfg.optimizer.top_n_similar
    
    # 临时缓存队列
    batch_size = cfg.optimizer.llm_batch_size
    batch_prompts = []
    batch_metadata = [] # 存元数据，用于回调更新: (rec_anchor, group_ids)

    for anchor_id in high_ids_sorted:
        if anchor_id not in memories: continue
        if anchor_id in merged_consumed_ids: continue

        rec_anchor = memories[anchor_id]
        cluster_id = rec_anchor.get("cluster_id")
        if cluster_id is None: continue
        cluster_id = int(cluster_id)
        
        cluster_member_ids = [str(x) for x in cluster_to_ids.get(cluster_id, [])]
        if not cluster_member_ids: continue

        candidates = [
            mid for mid in cluster_member_ids
            if mid != anchor_id and mid not in merged_consumed_ids
        ]
        if not candidates: continue

        anchor_emb = id_to_emb.get(anchor_id)
        if anchor_emb is None: continue

        sims = []
        for mid in candidates:
            emb = id_to_emb.get(mid)
            if emb is None: continue
            sims.append((mid, cosine_similarity(anchor_emb, emb)))

        if not sims: continue

        sims_sorted = sorted(sims, key=lambda x: -x[1])
        neighbors = [mid for mid, _ in sims_sorted[:top_n_similar]]
        group_ids = [anchor_id] + neighbors

        print(f"\n🔥 [Plan] Anchor {anchor_id} (freq={freq_map[anchor_id]}) 准备合并 top-{len(neighbors)} 邻居")
        print("   >>> 邻居详情 (ID | Similarity):")
        for mid, score in sims_sorted[:top_n_similar]:
             print(f"       - ID: {mid:<6} | Sim: {score:.4f}")
        # 🔥 关键修改 1: 立即标记邻居为“已消耗”，防止当前 Batch 后面的 Anchor 抢占
        # 虽然 LLM 还没跑完，但我们先占座，保证贪心逻辑的顺序性
        for mid in neighbors:
            merged_consumed_ids.add(mid)
            # 只有被合并且频次低的才删除
            if freq_map.get(mid, 0) < cfg.optimizer.low_freq_threshold:
                to_delete_ids.add(mid)
        
        # 构造 Prompt 文本
        group_texts = []
        for mid in group_ids:
            rec = memories[mid]
            text = rec.get("contents", "")
            group_texts.append(f"[ID {mid}] {text}")

        # 生成 Prompt 并加入 Batch 队列
        prompt = summarize_high_freq_prompt(group_texts, cfg)
        batch_prompts.append(prompt)
        batch_metadata.append({
            "rec_anchor": rec_anchor,
            "group_ids": group_ids,
            "anchor_id": anchor_id
        })

        # 🔥 关键修改 2: 凑够 Batch 立即执行
        if len(batch_prompts) >= batch_size:
            print(f"🚀 [Batch Execution] 并发执行 {len(batch_prompts)} 个高频聚合任务...")
            outputs = call_llm_batch(batch_prompts, cfg)
            
            # 回填结果
            for task_info, summary_text in zip(batch_metadata, outputs):
                if not summary_text:
                    print(f"   ⚠️ LLM 返回为空，跳过 Anchor {task_info['anchor_id']}")
                    continue
                
                rec = task_info['rec_anchor']
                rec["contents"] = summary_text
                rec["merged_from_ids"] = task_info['group_ids']
                rec["merge_type"] = "high_freq_anchor"
            
            # 清空队列
            batch_prompts = []
            batch_metadata = []

    # 🔥 关键修改 3: 处理循环结束后剩余的任务
    if batch_prompts:
        print(f"🚀 [Batch Execution] 处理剩余的 {len(batch_prompts)} 个高频聚合任务...")
        outputs = call_llm_batch(batch_prompts, cfg)
        for task_info, summary_text in zip(batch_metadata, outputs):
            if not summary_text: continue
            rec = task_info['rec_anchor']
            rec["contents"] = summary_text
            rec["merged_from_ids"] = task_info['group_ids']
            rec["merge_type"] = "high_freq_anchor"

    # 5. 低频：扩写
    print("\n========== 低频记忆扩写阶段 ==========")

    low_expand_ids = [
        mid for mid in low_ids
        if mid in memories and mid not in to_delete_ids
    ]
    print(f"🥶 需要扩写的低频记忆条目数: {len(low_expand_ids)}")

    low_expand_items = []
    for mid in low_expand_ids:
        rec = memories[mid]
        base_text = rec.get("contents", "")
        low_expand_items.append((mid, base_text))

    batch_size = cfg.optimizer.llm_batch_size
    total_low = len(low_expand_items)
    
    for start in range(0, total_low, batch_size):
        end = min(start + batch_size, total_low)
        batch_items = low_expand_items[start:end]
        batch_ids = [mid for (mid, _) in batch_items]

        print(f"\n🥶 扩写低频记忆 Batch {start // batch_size + 1} / { (total_low + batch_size - 1) // batch_size }")
        print(f"   IDs: {batch_ids}")

        # 🔥 修正: 这里之前漏传了 cfg 参数，现在补上
        batch_prompts = [
            expand_low_freq_memory_prompt(base_text, cfg) 
            for (_, base_text) in batch_items
        ]
        
        batch_outputs = call_llm_batch(batch_prompts, cfg)

        for (mid, base_text), expanded in zip(batch_items, batch_outputs):
            if not expanded:
                print(f"   ⚠️ LLM 返回为空，ID={mid} 保持原文不变")
                continue
            rec = memories[mid]
            rec["contents"] = expanded
            rec["opt_type"] = "low_freq_expanded"
    # 6. 写出新的记忆库
    print("\n========== 写出优化后的记忆库 ==========")
    kept_count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for mid in id_order:
            if mid not in memories: continue
            if mid in to_delete_ids: continue
            f.write(json.dumps(memories[mid], ensure_ascii=False) + "\n")
            kept_count += 1

    print(f"✅ 新记忆库写入完成: {output_file}")
    print(f"   保留记忆条目: {kept_count}")
    print(f"   删除记忆条目: {len(to_delete_ids)}")

if __name__ == "__main__":
    optimize_memory()