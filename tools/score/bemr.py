import math


# --- 新增 BEMR 理论辅助函数 ---

def _get_bemr_moments(alpha, beta):
    """
    根据 BEMR 理论 2.1 节计算一阶矩（期望）和二阶矩（方差）
    [cite: 1051, 1052]
    """
    total = alpha + beta
    # 一阶矩 (Expectation/Mean): 代表记忆的历史平均表现 [cite: 1051]
    mean = alpha / total
    
    # 二阶矩 (Variance): 代表记忆的不确定性 [cite: 1052]
    # Var = (αβ) / ((α+β)^2 * (α+β+1))
    variance = (alpha * beta) / ((total ** 2) * (total + 1))
    
    return mean, variance

def _calculate_bemr_final_score(alpha, beta, cfg):
    """
    根据 BEMR 理论 3.1 节公式 (4) 计算最终评分
    Score = λ1 * Mean + λ2 * Uncertainty_Bonus
    注意：这里去掉了 Sim(q,m) 项，因为这是一个全局离线打分函数，不是针对特定 Query 的检索打分
    """
    mean, variance = _get_bemr_moments(alpha, beta)
    
    # 从配置读取超参数，论文建议 lambda1=1.0, lambda2=0.1~0.5 
    lambda1 = cfg.parameters.get('bemr_lambda1', 1.0)
    lambda2 = cfg.parameters.get('bemr_lambda2', 0.5) 
    
    # UCB 评分策略：均值 + 探索奖励（基于方差/标准差）
    # 论文公式 (4) 中使用的是 1/sqrt(n)，本质是方差的变体。
    # 这里我们直接利用精确计算出的 variance。
    final_score = (lambda1 * mean) + (lambda2 * math.sqrt(variance))
    
    return final_score