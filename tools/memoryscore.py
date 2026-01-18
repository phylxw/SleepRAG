
import os
from omegaconf import DictConfig, OmegaConf
import json
from tqdm import tqdm
from tools.evaluate import judge_math_item
import matplotlib.pyplot as plt
from tools.score.bemr import _calculate_bemr_final_score
import copy


def _load_memory_corpus(corpus_file: str):
    """辅助函数：读取记忆库文件"""
    all_memory_ids = set()
    id_to_content = {} 
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                mid = str(item['id'])
                all_memory_ids.add(mid)
                id_to_content[mid] = item.get("contents", "")
    except Exception as e:
        print(f"⚠️ 无法读取记忆库文件 {corpus_file}，错误: {e}")
    return all_memory_ids, id_to_content

def _calculate_scores(rag_results, all_memory_ids, cfg, old_stats=None, baseline_scores=None):
    """
    修改版：支持 Counterfactual Update (差分更新 / 边际效用)
    """
    INIT_VAL = cfg.parameters.INIT_VAL
    # 1. 继承或初始化统计量 (不变)
    if old_stats:
        memory_stats = copy.deepcopy(old_stats)
        # 确保所有 memory_id 都在 stats 里
        for mid in all_memory_ids:
            if mid not in memory_stats:
                memory_stats[mid] = {'alpha': INIT_VAL, 'beta': INIT_VAL, 'pos_queries': [], 'neg_queries': []}
    else:
        memory_stats = {mid: {'alpha': INIT_VAL, 'beta': INIT_VAL, 'pos_queries': [], 'neg_queries': []} for mid in all_memory_ids}

    correct_count = 0
    
    # 2. 遍历结果更新状态 (注意：使用 enumerate 获取索引 i)
    for i, item in enumerate(tqdm(rag_results, desc="Scoring & Capturing Gradients (BEMR)")):
        
        # --- A. 获取 RAG 正确性 (With Memory) ---
        if cfg.experiment.tag in ["humaneval", "mbpp"]:
            is_rag_correct = (item.score == 1.0)
        else:
            try:
                is_rag_correct, _, _ = judge_math_item(item)
            except Exception:
                is_rag_correct = False
        
        if is_rag_correct: correct_count += 1

        # --- B. 🔥 获取 Baseline 正确性 (Without Memory) ---
        if baseline_scores and i < len(baseline_scores):
            # Baseline 的分数如果是 1.0 也就是对，0.0 是错
            is_base_correct = (baseline_scores[i] == 1.0)
        else:
            # 如果没有提供 Baseline (第一轮或被关掉)，为了安全：
            # 策略1: 假设 Baseline 全错 -> 退化回旧算法 (只要 RAG 对了就奖励)
            # 策略2: 假设 Baseline 全对 -> 极其保守 (除非 RAG 也是对的否则不奖励)
            # 这里选用策略1，保持兼容性
            is_base_correct = False 

        # --- C. 构造 TextGrad 用的 Query ---
        q = getattr(item, 'question', '') or getattr(item, 'prompt', '') or ''
        q = q.strip()
        gold_list = getattr(item, 'golden_answers', [])
        a = gold_list[0] if gold_list else "No Answer Provided"
        current_query = f"[Question]: {q}\n   [Target Answer]: {str(a)[:500]}"

        # --- D. 🔥 更新记忆权重 (核心逻辑) ---
        retrieved_docs = getattr(item, 'retrieval_result', [])
        
        for doc in retrieved_docs:
            doc_id = str(doc.get('id')) if isinstance(doc, dict) else str(getattr(doc, 'id', None))
            
            if doc_id and doc_id in memory_stats:
                
                # 🔥🔥🔥 [差分更新真值表] 🔥🔥🔥
                
                # Case 1: 雪中送炭 (Critical Success) [Base错 -> RAG对]
                # 这是最宝贵的记忆，大幅奖励
                if is_rag_correct and not is_base_correct:
                    memory_stats[doc_id]['alpha'] += 2.0  # 建议给 2.0 或更高，加速收敛
                    if current_query not in memory_stats[doc_id]['pos_queries']:
                        memory_stats[doc_id]['pos_queries'].append(current_query)
                
                # Case 2: 帮倒忙 (Toxic Failure) [Base对 -> RAG错]
                # 这是最有害的记忆，大幅惩罚
                elif not is_rag_correct and is_base_correct:
                    memory_stats[doc_id]['beta'] += 2.0   # 严厉惩罚
                    if current_query not in memory_stats[doc_id]['neg_queries']:
                        memory_stats[doc_id]['neg_queries'].append(current_query)
                
                # Case 3: 锦上添花 (Redundant) [Base对 -> RAG对]
                # 说明这题很简单，记忆可能有用也可能没用。
                # 给予微小奖励或不奖励，防止“万金油”记忆刷分
                elif is_rag_correct and is_base_correct:
                    memory_stats[doc_id]['alpha'] += 0.05  # 微小奖励，维持活跃度
                
                # Case 4: 无能为力 (Useless) [Base错 -> RAG错]
                # 记忆没起作用，但也没把本来对的搞错。
                # 给予中等惩罚，因为它占用了检索位但没解决问题
                elif not is_rag_correct and not is_base_correct:
                    memory_stats[doc_id]['beta'] += 0.25
                    # 也可以加入负样本队列，供 Expert 分析“为什么没帮上忙”
                    if current_query not in memory_stats[doc_id]['neg_queries']:
                        memory_stats[doc_id]['neg_queries'].append(current_query)

    # 5. 计算最终标量分数
    final_scores_map = {}
    for mid, stats in memory_stats.items():
        total = stats['alpha'] + stats['beta']
        # 计算 Beta 分布期望值
        score = stats['alpha'] / total if total > 0 else 0.5
        final_scores_map[mid] = score
    
    return final_scores_map, memory_stats, correct_count

def _print_stats_and_save(memory_scores, id_to_content, total_questions, correct_count, freq_file ,is_write = True):
    """辅助函数：打印统计信息并保存 JSONL 结果"""
    # 排序 (按分数从高到低)
    sorted_memories = sorted(memory_scores.items(), key=lambda x: (-x[1], x[0]))
    
    # 统计信息
    total_mem = len(sorted_memories)
    positive_mem = sum(1 for _, v in sorted_memories if v > 0.51)
    negative_mem = sum(1 for _, v in sorted_memories if v < 0.49)
    zero_mem = sum(1 for _, v in sorted_memories if v < 0.51 and v > 0.49)
    
    print(f"📊 记忆库评分统计:")
    print(f"   - 总量: {total_mem}")
    print(f"   - 正分(贡献者): {positive_mem} ({(positive_mem/total_mem)*100:.1f}%)")
    print(f"   - 负分(干扰项): {negative_mem} ({(negative_mem/total_mem)*100:.1f}%)")
    print(f"   - 零分(冷门): {zero_mem}")
    print(correct_count)
    print(total_questions)
    print(f"   - 当前题目正确率: {correct_count/total_questions*100:.2f}%")

    if is_write :
        # 导出 jsonl
        try:
            print(f"💾 [Save] 正在导出记忆评分结果到: {freq_file}")
            os.makedirs(os.path.dirname(freq_file), exist_ok=True)
            
            with open(freq_file, "w", encoding="utf-8") as f:
                for rank, (mid, score) in enumerate(sorted_memories, start=1):
                    record = {
                        "rank": rank,
                        "memory_id": mid,
                        "freq": round(score, 3), # 🔥 这里存的是分数
                        "contents": id_to_content.get(mid, "")
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("✅ 评分文件导出完成！")
        except Exception as e:
            print(f"❌ 导出失败: {e}")
        
    return sorted_memories

def _visualize_results(cfg: DictConfig, sorted_memories, vis_image_file: str):
    """辅助函数：生成分数分布图"""
    if cfg.experiment.visualize_memory:
        print(f"🎨 [Visual] 正在生成分数分布图: {vis_image_file}")
        try:
            ids = [m[0] for m in sorted_memories]
            scores = [m[1] for m in sorted_memories]
            
            display_limit = 30
            if len(ids) > display_limit * 2:
                plot_ids = ids[:display_limit] + ["..."] + ids[-display_limit:]
                plot_scores = scores[:display_limit] + [0] + scores[-display_limit:]
                # 颜色区分
                colors = []
                for s in plot_scores:
                    if s > 0: colors.append('skyblue')
                    elif s < 0: colors.append('salmon')
                    else: colors.append('lightgrey')
            else:
                plot_ids = ids
                plot_scores = scores
                colors = ['skyblue' if s > 0 else 'salmon' if s < 0 else 'lightgrey' for s in plot_scores]

            plt.figure(figsize=(15, 6))
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            bars = plt.bar(plot_ids, plot_scores, color=colors, edgecolor='navy')
            plt.title(f'Memory  Score', fontsize=14)
            plt.ylabel('Score')
            plt.xticks(rotation=90, fontsize=8) 
            
            # 显示数值
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if plot_ids[i] != "...": 
                    y_pos = height if height >= 0 else height - (max(scores)*0.05)
                    va = 'bottom' if height >= 0 else 'top'
                    plt.text(bar.get_x() + bar.get_width()/2., y_pos, f'{int(height*1000)/1000}',
                             ha='center', va=va, fontsize=8)
            
            plt.tight_layout()
            plt.savefig(vis_image_file, dpi=300)
            print("✅ 图片保存成功！")
        except ImportError:
            print("❌ 缺少 matplotlib")
    else:
        print("\n🏆 [Top 10 High-Utility Memories]")
        for mid, score in sorted_memories[:10]:
            print(f"   ID: {mid:<5} | Score: {score}")