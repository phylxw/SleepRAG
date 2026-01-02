
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

def _calculate_scores(rag_results, all_memory_ids, cfg: DictConfig, old_stats=None):
    """
    修改版：基于 BEMR (Bayesian-EM Memory Refinement) 计算分数
    功能：
    1. 继承上一轮状态 (持续学习)
    2. 更新 Alpha/Beta (贝叶斯更新)
    3. 捕获导致错误的 Query (作为 TextGrad 的梯度)
    """
    
    # 1. 继承或初始化统计量
    if old_stats:
        # 深拷贝以防修改原引用
        memory_stats = copy.deepcopy(old_stats)
        # 补齐可能新增的记忆 ID (防止 Key Error)
        for mid in all_memory_ids:
            if mid not in memory_stats:
                memory_stats[mid] = {'alpha': 1.0, 'beta': 1.0, 'pos_queries': [], 'neg_queries': []}
    else:
        # 冷启动：全部初始化为 Prior (1.0, 1.0)
        memory_stats = {mid: {'alpha': 1.0, 'beta': 1.0, 'pos_queries': [], 'neg_queries': []} for mid in all_memory_ids}

    correct_count = 0
    
    # 2. 遍历结果更新状态
    for item in tqdm(rag_results, desc="Scoring & Capturing Gradients (BEMR)"):
        # 假设 judge_math_item 在外部作用域可用
        is_correct, _, _ = judge_math_item(item)
        if is_correct: correct_count += 1

        # 获取当前 Query (这是 TextGrad 的“梯度”来源)
        current_query = getattr(item, 'question', '')

        retrieved_docs = getattr(item, 'retrieval_result', [])
        
        for doc in retrieved_docs:
            doc_id = str(doc.get('id')) if isinstance(doc, dict) else str(getattr(doc, 'id', None))
            
            # 只要 doc_id 存在于我们的库中，就进行更新
            if doc_id and doc_id in memory_stats:
                if is_correct:
                    # ✅ 答对：Alpha + 1
                    memory_stats[doc_id]['alpha'] += 1.0
                    # [E-Step] 记录正样本 (用于修正 Key)
                    if current_query and current_query not in memory_stats[doc_id]['pos_queries']:
                        memory_stats[doc_id]['pos_queries'].append(current_query)
                else:
                    # ❌ 答错：Beta + 1
                    memory_stats[doc_id]['beta'] += 1.0
                    # [TextGrad] 记录负样本 (用于修正 Content) -> 这就是梯度！
                    if current_query and current_query not in memory_stats[doc_id]['neg_queries']:
                        memory_stats[doc_id]['neg_queries'].append(current_query)

    # 3. 计算用于可视化的标量分数 (Mean Utility)
    # 注意：memory_stats 才是我们要存盘的核心数据，final_scores_map 只是给 print/vis 用的
    final_scores_map = {}
    for mid, stats in memory_stats.items():
        # 这里计算简单的均值用于热度展示: alpha / (alpha + beta)
        # 你也可以调用 _calculate_bemr_final_score 算 UCB 分数
        total = stats['alpha'] + stats['beta']
        score = stats['alpha'] / total if total > 0 else 0.5
        final_scores_map[mid] = score
    
    # 返回三个值：可视化分数表，完整的统计状态，正确数
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