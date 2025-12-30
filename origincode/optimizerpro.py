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
from utils.toolfunction import clean_special_chars, has_cuda
from tools.optimize.callllm import init_llm, call_llm, call_llm_batch
from tools.optimize.memoryload import load_clustered_memories, load_cluster_summary

# ================= 全局变量 =================
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_SGLANG_CLIENT = None

# ==========================================
# 1. Prompt 构造函数
# ==========================================

def textgrad_correction_prompt(content: str, neg_queries: List[str], good_examples: str, cfg: DictConfig) -> str:
    """
    TextGrad 核心 Prompt：结合错误反馈(Gradient)和正向示例(Momentum)来修正记忆
    """
    # 取前 3 个错误 Query 作为梯度信号
    neg_text = "\n".join([f"- {q}" for q in neg_queries[:3]])
    
    # 尝试从 config 读取模板，如果没有则使用默认硬编码模板
    default_template = """
You are optimizing a memory entry for a Retrieval-Augmented Generation (RAG) system.

[Original Memory]
{content}

[Critique / Gradient]
This memory was INCORRECTLY retrieved for the following queries (it misled the system):
{neg_text}

[Positive Guidance / Momentum]
Successful neighboring memories look like this (try to mimic their style/depth):
{good_examples}

[Task]
Rewrite the memory content. 
1. Make it SPECIFIC enough to avoid being retrieved for the incorrect queries above.
2. Maintain its core utility but clarify ambiguities.
3. If the memory contains factual errors, fix them based on common knowledge.

Output ONLY the rewritten memory content.
"""
    # 安全获取模板
    template = default_template
    if hasattr(cfg.optimizer, "prompts") and "textgrad_correction" in cfg.optimizer.prompts:
        template = cfg.optimizer.prompts.textgrad_correction
    
    prompt = template.format(content=content, neg_text=neg_text, good_examples=good_examples)
    return prompt

def summarize_experience_prompt(target_text: str, good_neighbors: List[str], cfg: DictConfig) -> str:
    """旧逻辑：利用高分邻居修正低分记忆 (Imitation)"""
    good_examples_text = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(good_neighbors))
    template = cfg.optimizer.prompts.expand_low_freq
    prompt = template.format(text=target_text, good_examples=good_examples_text)
    return prompt

def expand_low_freq_memory_prompt(text: str, good_examples: str, cfg: DictConfig) -> str:
    """旧逻辑：自我扩写 (Fallback)"""
    template = cfg.optimizer.prompts.expand_low_freq
    prompt = template.format(text=text, good_examples=good_examples)
    return prompt

# ==========================================
# 2. 筛选逻辑 (适配 BEMR Stats)
# ==========================================

def select_ids_from_stats(memory_stats: Dict[str, dict], cfg: DictConfig):
    """
    根据 BEMR Stats (Alpha/Beta) 筛选高分和低分记忆
    """
    # 计算胜率分数
    scores = []
    for mid, stats in memory_stats.items():
        alpha = stats.get('alpha', 1.0)
        beta = stats.get('beta', 1.0)
        total = alpha + beta
        # 计算胜率 (0.0 - 1.0)
        win_rate = alpha / total if total > 0 else 0.5
        scores.append((mid, win_rate, total))
    
    # 排序：按胜率降序，胜率相同按尝试次数降序
    scores.sort(key=lambda x: (-x[1], -x[2]))
    
    # 筛选
    top_k = cfg.optimizer.top_k_high
    bottom_k = cfg.optimizer.bottom_k_low
    
    high_ids = [x[0] for x in scores[:top_k]]
    
    # 低分：只选那些尝试过且失败过的 (win_rate < 0.4 且 total > 2)
    # 这种筛选能保证 TextGrad 有足够的“错误梯度”去优化
    bad_ids = [x[0] for x in scores if x[1] < 0.4 and x[2] > 2]
    
    # 如果没选够，就硬凑最后几个垫底的
    if len(bad_ids) < 10:
         bad_ids = [x[0] for x in scores[-bottom_k:]]

    print(f"🔥 高分 Anchor (用于指导): {len(high_ids)}")
    print(f"🥶 低分 Candidates (需要修正): {len(bad_ids)}")
    
    return set(high_ids), set(bad_ids)

# ==========================================
# 3. 主优化逻辑 (Hydra Managed)
# ==========================================

@hydra.main(version_base=None, config_path="conf", config_name="config")
def optimize_memory(cfg: DictConfig):
    # 0. 初始化 LLM
    init_llm(cfg)

    # 1. 读入路径 (🔥 使用 yaml 中定义的静态路径)
    # 你的 config.yaml 已经定义好了完整的路径，直接用即可
    cluster_file = cfg.paths.cluster_output
    summary_file = cfg.paths.cluster_summary
    stats_file = cfg.paths.stats_file       # 对应 ${experiment.tag}_memory_stats.json
    output_file = cfg.paths.optimized_memory # 对应 ${experiment.tag}_optimized_memory_topk.jsonl

    print(f"📂 [Input] 聚类结果: {cluster_file}")
    print(f"📂 [Input] 统计状态: {stats_file}")
    print(f"📂 [Output] 优化结果: {output_file}")

    # 加载数据
    if not os.path.exists(stats_file):
        print(f"❌ 找不到状态文件: {stats_file}，无法进行 TextGrad 优化！")
        return

    # 加载聚类数据
    memories, id_order = load_clustered_memories(cluster_file)
    cluster_to_ids = load_cluster_summary(summary_file)
    
    # 加载 BEMR 统计数据
    with open(stats_file, 'r', encoding='utf-8') as f:
        memory_stats = json.load(f)

    if not memories:
        print("❌ 无法加载记忆数据，程序退出。")
        return

    # 2. 筛选集合 (高分做老师，低分做学生)
    high_ids, bad_ids = select_ids_from_stats(memory_stats, cfg)

    # =========================================================
    # 4. 高频/高分优化 (Pruning: 优胜劣汰)
    # =========================================================
    print("\n========== 高分记忆清理阶段 (Pruning) ==========")
    to_delete_ids = set() 
    
    # 按 Cluster 分组
    cluster_groups = {}
    for mid, rec in memories.items():
        cid = rec.get("cluster_id")
        if cid is not None:
            cid = int(cid)
            if cid not in cluster_groups: cluster_groups[cid] = []
            cluster_groups[cid].append(mid)
    
    pruned_count = 0
    
    for cid, members in cluster_groups.items():
        if len(members) < 2: continue # 独生子不删
        
        # 获取该 Cluster 内所有成员的 Stats
        member_stats_list = []
        for mid in members:
            stats = memory_stats.get(mid, {'alpha': 1.0, 'beta': 1.0})
            total = stats['alpha'] + stats['beta']
            win_rate = stats['alpha'] / total if total > 0 else 0.5
            member_stats_list.append({
                'id': mid,
                'win_rate': win_rate,
                'total': total
            })
            
        # 找出该 Cluster 的“最强王者” (Anchor)
        member_stats_list.sort(key=lambda x: (-x['win_rate'], -x['total']))
        best_mem = member_stats_list[0]
        
        # 条件 A: 有强力 Anchor (胜率 > 0.7 且验证过)
        has_strong_anchor = (best_mem['win_rate'] > 0.7 and best_mem['total'] > 2)
        
        if has_strong_anchor:
            # 条件 B: 删除垃圾小弟
            for mem in member_stats_list[1:]:
                is_trash = False
                # 情况 1: 确实烂 (<40% 且不是冷启动)
                if mem['win_rate'] < 0.4 and mem['total'] > 2:
                    is_trash = True
                # 情况 2: 严重干扰 (Anchor > 90% 但小弟 < 50%)
                if best_mem['win_rate'] > 0.9 and mem['win_rate'] < 0.5:
                    is_trash = True
                
                if is_trash:
                    to_delete_ids.add(mem['id'])
                    pruned_count += 1

    print(f"✨ Pruning 完成，共删除了 {pruned_count} 条劣质冗余记忆。")

    # =========================================================
    # 5. TextGrad 核心修正阶段
    # =========================================================
    print("\n========== TextGrad 记忆修正阶段 (Gradient Descent) ==========")
    
    # 筛选需要处理的 ID (在 bad_ids 里且未被 Pruning 删除)
    low_expand_ids = [mid for mid in bad_ids if mid in memories and mid not in to_delete_ids]
    print(f"🎯 待优化目标数量: {len(low_expand_ids)}")

    batch_size = cfg.optimizer.llm_batch_size
    batch_prompts = []
    batch_metadata = [] 

    for mid in low_expand_ids:
        rec = memories[mid]
        base_text = rec.get("contents", "")
        cluster_id = rec.get("cluster_id")
        
        # 获取 BEMR Stats
        stats = memory_stats.get(mid, {})
        neg_queries = stats.get('neg_queries', []) # 🔥 错误反馈 (Gradient)
        
        # 寻找“优等生” (Momentum)
        good_neighbors_text = []
        if cluster_id is not None:
            cluster_id = int(cluster_id)
            members = cluster_to_ids.get(cluster_id, [])
            # 找同类里的高分 (Score > 0.8)
            for m_id in members:
                m_id = str(m_id)
                if m_id == mid: continue
                s = memory_stats.get(m_id, {})
                s_total = s.get('alpha', 0) + s.get('beta', 0)
                if s_total > 0 and (s.get('alpha', 0)/s_total) > 0.8:
                    good_neighbors_text.append(memories[m_id].get("contents", ""))
            
            # 取 Top 3
            good_neighbors_text = good_neighbors_text[:3]
        
        good_examples_str = "\n".join([f"- {t}" for t in good_neighbors_text])

        # 🔥 分支判断：优先 TextGrad
        if len(neg_queries) > 0:
            # Case A: TextGrad 修正 (最强)
            prompt = textgrad_correction_prompt(base_text, neg_queries, good_examples_str, cfg)
            opt_type = f"textgrad_with_{len(neg_queries)}_errors"
        elif good_neighbors_text:
            # Case B: 模仿优等生 (次选)
            prompt = summarize_experience_prompt(base_text, good_neighbors_text, cfg)
            opt_type = "neighbor_imitation"
        else:
            # Case C: 自我反思 (保底)
            prompt = expand_low_freq_memory_prompt(base_text, "", cfg)
            opt_type = "self_reflection"

        batch_prompts.append(prompt)
        batch_metadata.append({"mid": mid, "opt_type": opt_type})

        # 执行 Batch
        if len(batch_prompts) >= batch_size:
            print(f"🚀 [Batch] 处理 {len(batch_prompts)} 条 (含 TextGrad)...")
            outputs = call_llm_batch(batch_prompts, cfg)
            
            for meta, output_text in zip(batch_metadata, outputs):
                if output_text and len(output_text) > 10:
                    mid = meta['mid']
                    memories[mid]["contents"] = output_text
                    memories[mid]["opt_type"] = meta['opt_type']
            
            batch_prompts = []
            batch_metadata = []

    # 处理剩余 Batch
    if batch_prompts:
        print(f"🚀 [Batch] 处理剩余 {len(batch_prompts)} 条...")
        outputs = call_llm_batch(batch_prompts, cfg)
        for meta, output_text in zip(batch_metadata, outputs):
            if output_text and len(output_text) > 10:
                mid = meta['mid']
                memories[mid]["contents"] = output_text
                memories[mid]["opt_type"] = meta['opt_type']

    # 6. 写出结果
    print("\n========== 写出优化后的记忆库 ==========")
    kept_count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for mid in id_order:
            if mid not in memories: continue
            if mid in to_delete_ids: continue
            f.write(json.dumps(memories[mid], ensure_ascii=False) + "\n")
            kept_count += 1

    print(f"✅ 完成！优化后记忆库: {output_file}")
    print(f"   保留: {kept_count} | 删除: {len(to_delete_ids)}")

if __name__ == "__main__":
    optimize_memory()