import os
import json
from typing import Dict, List
import hydra
from omegaconf import DictConfig
import logging

# 🤫 日志降噪
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 假设你的文件结构如下，请确保 import 路径正确
from tools.optimize.callllm import init_llm, call_llm_batch
from tools.optimize.callexpert import init_expert_llm, call_expert, call_expert_batch
from tools.optimize.memoryload import load_clustered_memories, load_cluster_summary
from optimizeold.select import select_ids_from_stats
from optimizeold.prune import prune
# 一定要引用我们刚刚改好的新版 textgrad_opt
from optimizeold.textgrad_optpro import textgrad_opt 
from optimizeold.evolve import evolve_high_score_opt

@hydra.main(version_base=None, config_path="conf", config_name="config")
def optimize_memory(cfg: DictConfig):
    # =========================================================
    # 0. 初始化
    # =========================================================
    init_llm(cfg)          # 学生 (Qwen/DeepSeek)
    init_expert_llm(cfg)   # 专家 (Gemini/GPT4)

    # 1. 路径配置
    cluster_file = cfg.paths.cluster_output
    summary_file = cfg.paths.cluster_summary
    stats_file = cfg.paths.stats_file
    output_file = cfg.paths.optimized_memory
    stats_optimized_file = cfg.paths.stats_optimized_file

    # 2. 加载数据
    if not os.path.exists(stats_file):
        print(f"❌ 找不到状态文件: {stats_file}")
        return
    with open(stats_file, 'r', encoding='utf-8') as f:
        memory_stats = json.load(f)

    # memories: dict, id_order: list (旧的顺序)
    memories, id_order = load_clustered_memories(cluster_file)
    cluster_to_ids = load_cluster_summary(summary_file)
    
    if not memories: 
        print("❌ 记忆库加载为空，退出。")
        return

    # =========================================================
    # 3. 筛选 (Select)
    # =========================================================
    # high_ids: 高分记忆, bad_ids: 低分记忆
    high_ids, bad_ids = select_ids_from_stats(memory_stats, cfg)

    # =========================================================
    # 4. 剪枝 (Prune) - 标记要删除的 ID
    # =========================================================
    to_delete_ids = prune(memories, memory_stats)
    print(f"🗑️ 计划删除 {len(to_delete_ids)} 条冗余/无效记忆")

    # =========================================================
    # 4.5. 高分进化 (Evolve High Score) - 🔥 新增环节
    # =========================================================
    # 针对高分但有瑕疵的记忆，生成 SUPPLEMENT 或 SPLIT
    # 注意：这些新 ID 已经在 memory_stats 里初始化过了
    new_supplement_ids = evolve_high_score_opt(cfg, memories, memory_stats, high_ids)

    # =========================================================
    # 5. 低分优化 (TextGrad with Primitives)
    # =========================================================
    # 针对低分记忆进行修复、重写或扩展
    optimized_ids = textgrad_opt(cfg, memories, memory_stats, cluster_to_ids, bad_ids, to_delete_ids)

    # =========================================================
    # 6. 写出新记忆库 (Save)
    # =========================================================
    print("\n========== 写出优化后的记忆库 ==========")
    
    # 🔥 [Critical Fix] 修复新记忆丢失问题
    # 找出原有顺序里没有的新 ID (由 TextGrad EXPAND 和 Evolve SPLIT/SUPPLEMENT 产生)
    current_memory_ids = set(memories.keys())
    old_ids_set = set(id_order)
    new_ids = list(current_memory_ids - old_ids_set)
    
    if new_ids:
        print(f"✨ 检测到 {len(new_ids)} 条新增记忆 (Total New)，正在追加到保存列表...")
        # 将新 ID 追加到保存列表末尾
        final_save_order = id_order + new_ids
    else:
        final_save_order = id_order

    kept_count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for mid in final_save_order:
            # 1. 如果 ID 不在内存里 (可能加载时就丢了)，跳过
            if mid not in memories: continue
            # 2. 如果 ID 被标记删除了，跳过
            if mid in to_delete_ids: continue
            
            # 写入
            f.write(json.dumps(memories[mid], ensure_ascii=False) + "\n")
            kept_count += 1
            
    print(f"✅ 记忆库已保存: {output_file} (共 {kept_count} 条)")

    # =========================================================
    # 7. 状态同步 (Sync Stats)
    # =========================================================
    print("\n========== 同步 BEMR 状态 (Stats Sync) ==========")
    
    # 1. 物理删除：从 stats 中彻底移除被 pruned 的 ID
    for del_id in to_delete_ids:
        if del_id in memory_stats:
            del memory_stats[del_id]
            
    # 2. 状态重置：对本轮发生过变动的 ID (Refine/Replace/Expand/Supplement/Split)
    #    我们需要合并 optimized_ids (低分优化) 和 new_supplement_ids (高分进化)
    #    因为内容变了，旧的 alpha/beta 就不准了，需要重置为先验值
    
    # 🔥 [Fix] 合并两个集合
    all_changed_ids = optimized_ids.union(new_supplement_ids)
    
    for opt_id in all_changed_ids:
        if opt_id in memory_stats:
            memory_stats[opt_id]['alpha'] = 1.0
            memory_stats[opt_id]['beta'] = 1.0
            # 这里的 query 清空会在下面统一做，但也为了保险
            memory_stats[opt_id]['neg_queries'] = []
            memory_stats[opt_id]['pos_queries'] = []

    # 3. 全局清理：清理所有记忆的 Queries (为下一轮 Evaluation 腾空)
    #    但保留 未优化记忆 的 alpha/beta (历史战绩)
    cleaned_count = 0
    for mid in memory_stats:
        stats = memory_stats[mid]
        # 只清空 Query 列表，保留分数
        stats['pos_queries'] = []
        stats['neg_queries'] = []
        cleaned_count += 1
            
    print(f" 🗑️ 已物理移除 {len(to_delete_ids)} 条被删 Stats")
    print(f" 🔄 已重置 {len(all_changed_ids)} 条变动记忆的分数 (Low+High Opt)")
    print(f" ✨ 已清理 {cleaned_count} 条记忆的 Query 缓存")
    
    try:
        with open(stats_optimized_file, 'w', encoding='utf-8') as f:
            json.dump(memory_stats, f, ensure_ascii=False, indent=2)
        print(f"✅ [BEMR] 状态已同步: {stats_optimized_file}")
    except Exception as e:
        print(f"❌ 保存 Stats 失败: {e}")

if __name__ == "__main__":
    optimize_memory()