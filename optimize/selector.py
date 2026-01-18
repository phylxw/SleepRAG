from typing import Dict, List, Tuple, Set
from omegaconf import DictConfig, OmegaConf

def select_ids_from_stats(memory_stats: Dict[str, dict], cfg: DictConfig) -> Tuple[List[str], List[str], List[str]]:
    """
    Select IDs for the Tri-Stream Optimization Framework (ICML).
    
    Logic Flow:
    1. Pre-calculate metrics (WinRate, Friction, Obs).
    2. Stream 1 (Evolve): Pick High Friction items (High Alpha & High Beta).
    3. Stream 2 (High): Pick High WinRate items (High Alpha & Low Beta) - EXCLUDING Evolve IDs.
    4. Stream 3 (Bad): Pick Low WinRate items (High Beta).
    """
    INIT_VAL = cfg.parameters.INIT_VAL
    scores: List[dict] = []

    # ---- Config ----
    top_k_high = int(cfg.optimizer.get("top_k_high", 50))
    bottom_k_low = int(cfg.optimizer.get("bottom_k_low", 80))
    top_k_evolve = int(cfg.optimizer.get("top_k_evolve", 50))

    # Thresholds
    legacy_freq_th = float(cfg.optimizer.get("low_freq_threshold", 1))
    min_obs = float(cfg.optimizer.get("min_obs_threshold", legacy_freq_th))
    evolve_win_rate_th = float(cfg.optimizer.get("evolve_win_rate_threshold", 0.5))

    # =========================================================
    # 0. Pre-calculation (一次性算好所有指标)
    # =========================================================
    for mid, stats in memory_stats.items():
        alpha = float(stats.get("alpha", INIT_VAL))
        beta = float(stats.get("beta", INIT_VAL))
        total = alpha + beta
        
        # 避免除以零
        win_rate = alpha / total if total > 1e-6 else 0.5
        
        # 计算有效观测数 (去除初始值的影响)
        n_obs = max(0.0, total - (INIT_VAL * 2))
        
        # 计算 Friction (摩擦力/争议度)
        # 只有当 Alpha 和 Beta 都很大时，Friction 才会大
        # 这里的摩擦力公式：(Alpha * Beta) / Total^2 (归一化到 0-0.25) 或者 (Alpha * Beta) / Total
        # 用 simplified harmonic mean 变体:
        friction = (alpha * beta) / total if total > 1e-6 else 0.0

        scores.append({
            "mid": str(mid),
            "win_rate": win_rate,
            "n_obs": n_obs,
            "alpha": alpha,
            "beta": beta,
            "total": total,
            "friction": friction,
            "neg_len": len(stats.get("neg_queries", []))
        })

    # =========================================================
    # Stream 1: Evolution Candidates (优先挑选！)
    # =========================================================
    # 定义：总体是好的(WinRate >= 0.5)，但存在严重争议(High Friction/Beta)
    evolve_candidates = []
    for s in scores:
        # 1. 过滤：只看胜率过得去的（太差的直接去 Bad Stream 了）
        if s["win_rate"] < evolve_win_rate_th:
            continue
        
        # 2. 过滤：活跃度门槛
        if s["n_obs"] < min_obs:
            continue

        # 3. 核心过滤：必须有“痛苦经历” (Beta 显著)
        # 如果 Beta 还没超过初始值太多，说明没怎么错过，不需要进化
        # 比如 INIT=1, Beta必须 > 1.5 或 2.0 才算有摩擦
        if s["beta"] <= (INIT_VAL + 0.5): 
            continue
            
        evolve_candidates.append(s)
    
    # 排序：摩擦力最大的优先 (说明模型对此最困惑)
    # Secondary Sort: 负样本数量 (越多越好分析)
    evolve_candidates.sort(key=lambda x: (-x["friction"], -x["neg_len"]))
    
    # 截断
    evolve_final = evolve_candidates[:top_k_evolve]
    evolve_ids = [x["mid"] for x in evolve_final]
    evolve_ids_set = set(evolve_ids) # 方便后续 O(1) 查找

    # =========================================================
    # Stream 2: High-Score Retention (捡剩下的好果子)
    # =========================================================
    # 定义：胜率高，且非常纯粹 (低摩擦)，且没被 Evolve 选走
    high_candidates = []
    for s in scores:
        # 1. 过滤：必须是赢家
        if s["win_rate"] < 0.5:
            continue
            
        # 2. 过滤：活跃度
        if s["n_obs"] < min_obs:
            continue

        # 3. 【关键互斥】：如果已经被选去进化了，这里就不要了
        if s["mid"] in evolve_ids_set:
            continue
            
        high_candidates.append(s)

    # 排序：胜率高的优先，胜率一样看 Alpha (绝对贡献)
    high_candidates.sort(key=lambda x: (-x["win_rate"], -x["alpha"]))
    
    # 截断
    high_final = high_candidates[:top_k_high]
    high_ids = [x["mid"] for x in high_final]

    # =========================================================
    # Stream 3: Low-Score Restoration (独立筛选)
    # =========================================================
    # 定义：胜率低的“垃圾”记忆
    bad_candidates = []
    for s in scores:
        # 1. 过滤：输家
        if s["win_rate"] >= 0.5:
            continue
            
        # 2. 过滤：活跃度 (注意：这里可能需要宽容一点，或者由外部 Config 控制)
        # 如果一个记忆只错了一次(Case 4)，Beta稍涨，WinRate下降，应该被捕捉
        if s["n_obs"] < min_obs:
            continue
            
        bad_candidates.append(s)

    # 排序：
    # 第一优先级: WinRate 越低越好 (升序) -> 0.1 比 0.4 更急需修复
    # 第二优先级: Total (活跃度) 越高越好 (降序) -> 同样是 0.1 胜率，错 100 次的比错 1 次的危害更大！
    bad_candidates.sort(key=lambda x: (x["win_rate"], -x["total"]))
    
    # 截断
    bad_final = bad_candidates[:bottom_k_low]
    bad_ids = [x["mid"] for x in bad_final]

    # ---- 打印调试信息 (方便你看到每个流选了啥) ----
    print(f"\n📊 [Tri-Stream Selection Report]")
    
    print(f" 🧬 Evolution Stream (Top {len(evolve_ids)}) | Criteria: High Friction")
    if evolve_final:
        print(f"    Sample: ID={evolve_final[0]['mid']} | Win={evolve_final[0]['win_rate']:.2f} | Beta={evolve_final[0]['beta']:.1f} | Fric={evolve_final[0]['friction']:.2f}")
    else:
        print("    [Empty] No candidates met criteria.")

    print(f" 🔹 Retention Stream (Top {len(high_ids)}) | Criteria: High WinRate")
    if high_final:
        print(f"    Sample: ID={high_final[0]['mid']} | Win={high_final[0]['win_rate']:.2f} | Alpha={high_final[0]['alpha']:.1f}")

    print(f" 🔸 Restoration Stream (Top {len(bad_ids)}) | Criteria: Low WinRate")
    if bad_final:
        print(f"    Sample: ID={bad_final[0]['mid']} | Win={bad_final[0]['win_rate']:.2f} | Total={bad_final[0]['total']:.1f}")

    return high_ids, bad_ids, evolve_ids