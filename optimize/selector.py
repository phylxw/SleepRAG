from typing import Dict, List, Tuple
from omegaconf import DictConfig, OmegaConf

def select_ids_from_stats(memory_stats: Dict[str, dict], cfg: DictConfig) -> Tuple[List[str], List[str], List[str]]:
    """
    Select IDs for the Tri-Stream Optimization Framework (ICML).
    
    Returns three distinct lists:
    1. high_ids (Retention/Pruning): Top WinRate. Used for pruning redundancy.
    2. bad_ids  (Correction): Bottom WinRate. Used for TextGrad repair.
    3. evolve_ids (Evolution): High Beta (Friction). Used for Split/Supplement.
    """
    scores: List[dict] = []

    # ---- Config ----
    # 数量限制
    top_k_high = int(cfg.optimizer.get("top_k_high", 50))       # 剪枝候选池大小
    bottom_k_low = int(cfg.optimizer.get("bottom_k_low", 80))   # 低分修复池大小
    top_k_evolve = int(cfg.optimizer.get("top_k_evolve", 50))   # 进化候选池大小 (新增)

    # 门槛
    legacy_freq_th = float(cfg.optimizer.get("low_freq_threshold", 1))
    min_obs = float(cfg.optimizer.get("min_obs_threshold", legacy_freq_th))
    
    # 进化流的特殊门槛：必须是“好人” (Win >= 0.5) 才能进化，坏人直接去修了
    evolve_win_rate_th = float(cfg.optimizer.get("evolve_win_rate_threshold", 0.5))

    for mid, stats in memory_stats.items():
        alpha = float(stats.get("alpha", 1.0))
        beta = float(stats.get("beta", 1.0))
        total = alpha + beta
        win_rate = alpha / total if total > 0 else 0.5
        n_obs = max(0.0, total - 2.0)
        neg_q_len = len(stats.get("neg_queries", []))

        scores.append({
            "mid": str(mid),
            "win_rate": win_rate,
            "n_obs": n_obs,
            "alpha": alpha,
            "beta": beta,
            "total": total,
            "neg_len": neg_q_len
        })

    # =========================================================
    # Stream 1: High-Score Retention (用于剪枝/维护)
    # =========================================================
    # 逻辑：谁最完美谁排前面 (WinRate Desc, Alpha Desc)
    # 目的：找出最强的记忆，后续 Prune 模块会看这些记忆是否语义重复，保留最强的
    high_pool = [s for s in scores if s["n_obs"] >= min_obs and s["win_rate"] >= 0.5]
    high_pool.sort(key=lambda x: (-x["win_rate"], -x["alpha"], x["mid"]))
    high_ids = [x["mid"] for x in high_pool[:top_k_high]]

    # =========================================================
    # Stream 2: Low-Score Restoration (用于修复/重写)
    # =========================================================
    # 逻辑：谁最烂谁排前面 (WinRate Asc)
    # 目的：找出拖后腿的，送去 TextGrad (Refine/Replace)
    low_pool = [s for s in scores if s["n_obs"] >= min_obs and s["win_rate"] < 0.5]
    low_pool.sort(key=lambda x: (x["win_rate"], -x["n_obs"], x["mid"]))
    bad_ids = [x["mid"] for x in low_pool[:bottom_k_low]]

    # =========================================================
    # Stream 3: Evolution Candidates (用于进化/细分)
    # =========================================================
    # 逻辑：在好人堆里(Win>=0.5)，谁的摩擦(Beta)最大，谁排前面
    # 目的：找出有争议的“高分”，送去 Expert (Supplement/Split)
    evolve_pool = [
        s for s in scores 
        if (s["win_rate"] >= evolve_win_rate_th)  # 必须是“总体正确”的
        and (s["beta"] > 1.0)                     # 必须有过失败经历 (Beta>1代表只要有错题)
        and (s["mid"] not in bad_ids)             # 互斥：不能是已经被划为烂记忆的
    ]
    
    # 排序核心：Beta 越大 -> 错得越多 -> 进化需求越强
    evolve_pool.sort(key=lambda x: (-x["beta"], -x["neg_len"], -x["n_obs"]))
    evolve_ids = [x["mid"] for x in evolve_pool[:top_k_evolve]]

    # ---- 打印统计 ----
    print(f"\n📊 [Tri-Stream Selection]")
    print(f"   🔹 Retention Stream (High IDs) : {len(high_ids)} (Sort: WinRate Desc)")
    print(f"   🔸 Restoration Stream (Bad IDs): {len(bad_ids)}  (Sort: WinRate Asc)")
    print(f"   🧬 Evolution Stream (Evolve IDs): {len(evolve_ids)} (Sort: Beta Desc)")

    return high_ids, bad_ids, evolve_ids


# ==============================================================================
# 🧪 测试代码 (Run this file directly)
# ==============================================================================
if __name__ == "__main__":
    # Mock Data: 模拟真实的 ICML 实验数据分布
    mock_stats = {
        # 1. 完美记忆 (Retention Candidates)
        "mem_perfect_1": {"alpha": 100, "beta": 0, "neg_queries": []},
        "mem_perfect_2": {"alpha": 50, "beta": 0, "neg_queries": []},

        # 2. 摩擦记忆 (Evolution Candidates) - 总体是好的，但经常在特定Case出错
        "mem_friction_high": {"alpha": 80, "beta": 20, "neg_queries": ["err"]*20}, # Win=0.8, Beta=20
        "mem_friction_mid":  {"alpha": 90, "beta": 5, "neg_queries": ["err"]*5},  # Win=0.9, Beta=5
        
        # 3. 垃圾记忆 (Restoration Candidates)
        "mem_trash_1": {"alpha": 1, "beta": 50, "neg_queries": ["err"]*50}, # Win=0.02
        "mem_trash_2": {"alpha": 10, "beta": 20, "neg_queries": ["err"]*20}, # Win=0.33
        
        # 4. 新记忆 (0.5分)
        "mem_new": {"alpha": 1, "beta": 1, "neg_queries": []},
    }

    # Config
    cfg = OmegaConf.create({
        "optimizer": {
            "top_k_high": 5,
            "bottom_k_low": 5,
            "top_k_evolve": 5, # 新增参数
            "min_obs_threshold": 1,
            "evolve_win_rate_threshold": 0.5
        }
    })

    print("🚀 Running Tri-Stream Selection Test...\n")
    high_ids, bad_ids, evolve_ids = select_ids_from_stats(mock_stats, cfg)

    # 验证 High (剪枝流)
    print("-" * 60)
    print(f"🔹 Retention Stream (High IDs) | 预期: 完美的高分记忆")
    print("-" * 60)
    for mid in high_ids:
        s = mock_stats[mid]
        total = s['alpha'] + s['beta']
        wr = s['alpha'] / total
        print(f"ID: {mid:<20} | Win: {wr:.2f} | Alpha: {s['alpha']}")

    # 验证 Bad (修复流)
    print("\n" + "-" * 60)
    print(f"🔸 Restoration Stream (Bad IDs) | 预期: 胜率最低的")
    print("-" * 60)
    for mid in bad_ids:
        s = mock_stats[mid]
        total = s['alpha'] + s['beta']
        wr = s['alpha'] / total
        print(f"ID: {mid:<20} | Win: {wr:.2f} | Alpha: {s['alpha']}")

    # 验证 Evolve (进化流)
    print("\n" + "-" * 60)
    print(f"🧬 Evolution Stream (Evolve IDs) | 预期: 高Beta的'好'记忆")
    print("-" * 60)
    for mid in evolve_ids:
        s = mock_stats[mid]
        total = s['alpha'] + s['beta']
        wr = s['alpha'] / total
        print(f"ID: {mid:<20} | Beta: {s['beta']:<4} | Win: {wr:.2f} (Needs Split/Supp)")