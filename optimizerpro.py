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

def expand_low_freq_memory_prompt(text: str, good_examples: str, cfg: DictConfig) -> str:
    """构造低频记忆扩写的 prompt"""
    template = cfg.optimizer.prompts.expand_low_freq
    prompt = template.format(text=text,good_examples = good_examples)
    
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
    high_ids = []
    for mid, f in sorted_desc:
        if f < 2: 
            # 一旦分数掉到 2 以下，后面的都不看了，直接截断
            break
        high_ids.append(mid)
        if len(high_ids) >= top_k_high:
            break

    # 低频：按 freq 升序
    sorted_asc = sorted(items, key=lambda x: x[1])
    bad_ids = []
    for mid, f in sorted_asc:
        if f <= -1:
            bad_ids.append(mid)

    print(f"🔥 高频 anchor 数量: {len(high_ids)}")
    print(f"🧊 分数小于-1的记忆数量: {len(bad_ids)}（之后会修正）")
    return set(high_ids), set(bad_ids)

def summarize_experience_prompt(target_text: str, good_neighbors: List[str], cfg: DictConfig) -> str:
    """构造利用高分邻居修正低分记忆的 Prompt"""
    good_examples_text = "\n".join(
        f"[{i+1}] {t}" for i, t in enumerate(good_neighbors)
    )
    template = cfg.optimizer.prompts.expand_low_freq
    prompt = template.format(text=target_text, good_examples=good_examples_text)
    return prompt

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
    high_ids, bad_ids = select_high_low_ids(
        freq_map,
        top_k_high=cfg.optimizer.top_k_high,
        bottom_k_low=cfg.optimizer.bottom_k_low,
        low_freq_for_low_only=cfg.optimizer.low_freq_threshold
    )

    # 3. 准备向量
    id_to_emb = build_embeddings_for_memories(memories, cfg.model.embedding_name)

    # 4. 高频：类内清理（Pruning）—— 仅删除低分邻居，不调用 LLM
    print("\n========== 高频记忆优化阶段 (Pruning Only: Delete Low Score Neighbors) ==========")
    to_delete_ids = set()

    # 按照频次从高到低排序，优先处理高分 Anchor
    high_ids_sorted = sorted(list(high_ids), key=lambda x: -freq_map.get(x, 0))
    
    count_pruned = 0

    for anchor_id in high_ids_sorted:
        if anchor_id not in memories: continue
        # 如果 Anchor 自己本身就在删除列表里（虽然逻辑上高分不应该在），跳过
        if anchor_id in to_delete_ids: continue

        rec_anchor = memories[anchor_id]
        cluster_id = rec_anchor.get("cluster_id")
        if cluster_id is None: continue
        cluster_id = int(cluster_id)
        
        # 获取同 Cluster 的所有成员
        cluster_member_ids = [str(x) for x in cluster_to_ids.get(cluster_id, [])]
        if not cluster_member_ids: continue

        # 筛选出需要“清理”的邻居
        # 条件：
        # 1. 不是 Anchor 自己
        # 2. 还没被标记删除
        # 3. 分数 < 1 (根据你的要求：分数小于1的全部删掉)
        victims = []
        for mid in cluster_member_ids:
            if mid == anchor_id: continue
            if mid in to_delete_ids: continue
            
            # 获取该邻居的分数，默认为 0
            score = freq_map.get(mid, 0)
            
            if score < 1:
                victims.append(mid)
        
        if not victims: continue

        print(f"🔥 [Pruning] Anchor {anchor_id} (Score={freq_map[anchor_id]}) 所在 Cluster {cluster_id} 清理:")
        print(f"   >>> 删除 {len(victims)} 个低分邻居 (Score < 1)")
        
        # 执行删除标记
        for mid in victims:
            to_delete_ids.add(mid)
            count_pruned += 1
            # 只有少量删除时可以打印出来看看，太多就不打印了
            if len(victims) <= 50:
                print(f"       - 🗑️ Delete ID: {mid:<6} (Score: {freq_map.get(mid, 0)})")

    print(f"\n✨ 高频优化阶段结束，共清理了 {count_pruned} 条低分冗余记忆。")
    # 注意：这里不再有 batch_prompts 或 call_llm_batch 的逻辑了

# 5. 低频/负分：利用类内高分“优等生”进行修正
    print("\n========== 低频/负分记忆修正阶段 (Correct with Top-5 Neighbors) ==========")

    # 筛选需要处理的低分记忆 (在 memories 中且未被删除)
    low_expand_ids = [
        mid for mid in bad_ids
        if mid in memories and mid not in to_delete_ids
    ]
    print(f"🥶 需要修正的低频/负分记忆条目数: {len(low_expand_ids)}")

    batch_size = cfg.optimizer.llm_batch_size
    batch_prompts = []
    batch_metadata = [] # 存元数据: (mid, 原文, 是否使用了邻居修正)

    for mid in low_expand_ids:
        rec = memories[mid]
        base_text = rec.get("contents", "")
        cluster_id = rec.get("cluster_id")
        
        # 1. 尝试寻找类内的高分“优等生”
        good_neighbors_text = []
        if cluster_id is not None:
            cluster_id = int(cluster_id)
            members = cluster_to_ids.get(cluster_id, [])
            
            # 筛选条件：Score >= 2 且不是自己
            candidates = []
            for m_id in members:
                m_id = str(m_id)
                if m_id == mid: continue
                if freq_map.get(m_id, 0) >= 2: # 🔥 核心条件：只学好的
                    candidates.append(m_id)
            
            # 取 Top-5 (按分数降序)
            candidates_sorted = sorted(candidates, key=lambda x: -freq_map.get(x, 0))
            top_k_candidates = candidates_sorted[:5]
            
            # 获取文本
            for m_id in top_k_candidates:
                if m_id in memories:
                    good_neighbors_text.append(memories[m_id].get("contents", ""))

        # 2. 根据是否找到“优等生”构建 Prompt
        if good_neighbors_text:
            # Plan A: 有优等生带飞 -> 结合 Top-5 修正
            prompt = summarize_experience_prompt(base_text, good_neighbors_text, cfg)
            use_neighbors = True
        else:
            # Plan B: 整个聚类都只有它自己或都很烂 -> 只能自己自我反思/扩写 (兜底)
            prompt = expand_low_freq_memory_prompt(base_text, good_examples = '' , cfg = cfg)
            use_neighbors = False
            
        batch_prompts.append(prompt)
        batch_metadata.append({
            "mid": mid,
            "use_neighbors": use_neighbors,
            "neighbor_count": len(good_neighbors_text)
        })

        # 3. 凑够 Batch 执行
        if len(batch_prompts) >= batch_size:
            print(f"🚀 [Batch Execution] 处理 {len(batch_prompts)} 条低分记忆...")
            outputs = call_llm_batch(batch_prompts, cfg)
            
            for meta, output_text in zip(batch_metadata, outputs):
                mid = meta['mid']
                if not output_text:
                    print(f"   ⚠️ LLM 返回为空，ID={mid} 保持不变")
                    continue
                
                rec = memories[mid]
                rec["contents"] = output_text
                
                if meta['use_neighbors']:
                    rec["opt_type"] = f"corrected_by_{meta['neighbor_count']}_neighbors"
                    # 可以在日志里标记一下
                    # print(f"   ✅ ID {mid} 已利用 {meta['neighbor_count']} 个高分邻居修正")
                else:
                    rec["opt_type"] = "self_expanded_fallback"

            batch_prompts = []
            batch_metadata = []

    # 处理剩余的
    if batch_prompts:
        print(f"🚀 [Batch Execution] 处理剩余 {len(batch_prompts)} 条低分记忆...")
        outputs = call_llm_batch(batch_prompts, cfg)
        for meta, output_text in zip(batch_metadata, outputs):
            if not output_text: continue
            rec = memories[meta['mid']]
            rec["contents"] = output_text
            rec["opt_type"] = f"corrected_by_{meta['neighbor_count']}_neighbors" if meta['use_neighbors'] else "self_expanded_fallback"
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