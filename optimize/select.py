from typing import Dict, List, Set, Tuple
import hydra
from omegaconf import DictConfig

def select_ids_from_stats(memory_stats: Dict[str, dict], cfg: DictConfig) -> Tuple[Set[str], Set[str]]:
    scores = []
    
    # 1. 获取配置参数
    top_k = cfg.optimizer.get("top_k_high", 50)
    bottom_k = cfg.optimizer.get("bottom_k_low", 80)
    # 频率阈值：只有访问次数超过这个值的才参与评估
    # 避免刚生成还没用过几次的记忆被误删或误优化
    freq_threshold = cfg.optimizer.get("low_freq_threshold", 1)

    # 2. 计算所有记忆的分数
    for mid, stats in memory_stats.items():
        alpha = stats.get('alpha', 1.0)
        beta = stats.get('beta', 1.0)
        total = alpha + beta
        
        # 简单平滑处理，避免 total=0
        win_rate = alpha / total if total > 0 else 0.5
        scores.append({
            "mid": mid,
            "win_rate": win_rate,
            "total": total
        })
    
    # -------------------------------------------------------
    # 3. 筛选 High IDs (优等生 - 用于 Momentum / 榜样)
    # -------------------------------------------------------
    # 排序规则：胜率从高到低，访问量从多到少
    scores.sort(key=lambda x: (-x["win_rate"], -x["total"]))
    
    high_ids = [x["mid"] for x in scores[:top_k]]

    # -------------------------------------------------------
    # 4. 筛选 Bad IDs (差生 - 用于 TextGrad 优化)
    # -------------------------------------------------------
    # 策略：
    # A. 过滤掉“太新”的记忆 (total <= threshold)，给新记忆一点机会
    # B. 过滤掉“表现还行”的记忆 (win_rate >= 0.5)，只优化不及格的
    candidates = [
        x for x in scores 
        if x["total"] > freq_threshold and x["win_rate"] < 0.5
    ]

    # C. 重新排序：我们希望最先优化“最烂”的
    # 排序规则：
    # 1. 胜率越低越优先 (x["win_rate"] 升序)
    # 2. 如果胜率一样(比如都是0)，错得越多越优先 (-x["total"] 降序，即 total 越大越前)
    candidates.sort(key=lambda x: (x["win_rate"], -x["total"]))

    # D. 🔥 [关键修改] 严格截断，不超过 bottom_k
    bad_ids = [x["mid"] for x in candidates[:bottom_k]]

    # -------------------------------------------------------
    # 5. 打印统计信息
    # -------------------------------------------------------
    print(f"📊 [Select Stats]")
    print(f"   - 总记忆数: {len(scores)}")
    print(f"   - 候选差生数(Candidates): {len(candidates)}")
    print(f"   - 限制阈值(Bottom K): {bottom_k}")
    print(f"🔥 最终产出 High IDs: {len(high_ids)}")
    print(f"🥶 最终产出 Bad IDs : {len(bad_ids)}")
    
    return set(high_ids), set(bad_ids)