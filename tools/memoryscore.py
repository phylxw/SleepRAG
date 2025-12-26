
import os
from omegaconf import DictConfig, OmegaConf
import json
from tqdm import tqdm
from tools.evaluate import judge_math_item
import matplotlib.pyplot as plt
from tools.score.bemr import _calculate_bemr_final_score

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

def _calculate_scores(rag_results, all_memory_ids, cfg: DictConfig):
    """
    修改版：基于 BEMR (Bayesian-EM Memory Refinement) 计算记忆分数
    [cite: 1040]
    """
    # 1. 初始化统计量：alpha(正例), beta(负例)
    # 论文建议初始化为 1 (Prior)，避免冷启动时的除零错误 
    memory_stats = {mid: {'alpha': 1.0, 'beta': 1.0} for mid in all_memory_ids}
    correct_count = 0
    
    # 2. 遍历结果更新 Alpha/Beta (E-Step 的数据收集部分)
    for item in tqdm(rag_results, desc="Scoring Memories (BEMR)"):
        # 假设 judge_math_item 在外部作用域可用
        is_correct, _, _ = judge_math_item(item)
        if is_correct:
            correct_count += 1

        retrieved_docs = getattr(item, 'retrieval_result', [])
        
        for doc in retrieved_docs:
            doc_id = str(doc.get('id')) if isinstance(doc, dict) else str(getattr(doc, 'id', None))
            
            # 只要 doc_id 存在于我们的库中，就进行贝叶斯更新
            if doc_id and doc_id in memory_stats:
                if is_correct:
                    # 答对：增加 alpha 
                    # 如果你想保留 cfg.experiment.reward 的权重控制，可以乘在 1 上，但标准 BEMR 是计数
                    memory_stats[doc_id]['alpha'] += 1.0 
                else:
                    # 答错：增加 beta
                    memory_stats[doc_id]['beta'] += 1.0

    # 3. 计算最终 BEMR 分数 (M-Step 准备阶段)
    memory_scores = {}
    for mid, stats in memory_stats.items():
        # 调用辅助函数计算混合分数
        score = _calculate_bemr_final_score(stats['alpha'], stats['beta'], cfg)
        memory_scores[mid] = score
    
    return memory_scores, correct_count

def _print_stats_and_save(memory_scores, id_to_content, total_questions, correct_count, freq_file):
    """辅助函数：打印统计信息并保存 JSONL 结果"""
    # 排序 (按分数从高到低)
    sorted_memories = sorted(memory_scores.items(), key=lambda x: (-x[1], x[0]))
    
    # 统计信息
    total_mem = len(sorted_memories)
    positive_mem = sum(1 for _, v in sorted_memories if v > 0)
    negative_mem = sum(1 for _, v in sorted_memories if v < 0)
    zero_mem = sum(1 for _, v in sorted_memories if v == 0)
    
    print(f"📊 记忆库评分统计:")
    print(f"   - 总量: {total_mem}")
    print(f"   - 正分(贡献者): {positive_mem} ({(positive_mem/total_mem)*100:.1f}%)")
    print(f"   - 负分(干扰项): {negative_mem} ({(negative_mem/total_mem)*100:.1f}%)")
    print(f"   - 零分(冷门): {zero_mem}")
    print(correct_count)
    print(total_questions)
    print(f"   - 当前题目正确率: {correct_count/total_questions*100:.2f}%")

    # 导出 jsonl
    try:
        print(f"💾 [Save] 正在导出记忆评分结果到: {freq_file}")
        os.makedirs(os.path.dirname(freq_file), exist_ok=True)
        
        with open(freq_file, "w", encoding="utf-8") as f:
            for rank, (mid, score) in enumerate(sorted_memories, start=1):
                record = {
                    "rank": rank,
                    "memory_id": mid,
                    "freq": int(score), # 🔥 这里存的是分数
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
            plt.title(f'Memory Utility Score (Correct=+2, Wrong=-2)', fontsize=14)
            plt.ylabel('Score')
            plt.xticks(rotation=90, fontsize=8) 
            
            # 显示数值
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if plot_ids[i] != "...": 
                    y_pos = height if height >= 0 else height - (max(scores)*0.05)
                    va = 'bottom' if height >= 0 else 'top'
                    plt.text(bar.get_x() + bar.get_width()/2., y_pos, f'{int(height)}',
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