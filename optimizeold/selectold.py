from typing import Dict, List, Tuple
from omegaconf import DictConfig


def select_ids_from_stats(memory_stats: Dict[str, dict], cfg: DictConfig) -> Tuple[List[str], List[str]]:
    """Select high-score and low-score memory IDs in a **stable, deterministic** order.

    Key fixes vs. typical buggy selectors:
    - Uses *observed count* n_obs = alpha + beta - 2 (treating (1,1) as prior) to avoid misclassifying new memories.
    - Returns **lists** (not sets) so downstream batching is reproducible.
    """
    scores: List[dict] = []

    # ---- Config (with backward-compatible fallbacks) ----
    top_k = int(cfg.optimizer.get("top_k_high", 50))
    bottom_k = int(cfg.optimizer.get("bottom_k_low", 80))

    # Old code used low_freq_threshold on (alpha+beta). We map that to observed count.
    legacy_freq_th = float(cfg.optimizer.get("low_freq_threshold", 1))
    min_obs = float(cfg.optimizer.get("min_obs_threshold", legacy_freq_th))

    high_win_rate_th = float(cfg.optimizer.get("high_win_rate_threshold", 0.5))
    low_win_rate_th = float(cfg.optimizer.get("low_win_rate_threshold", 0.5))  # old behavior used 0.5

    # ---- Score each memory ----
    for mid, stats in memory_stats.items():
        alpha = float(stats.get("alpha", 1.0))
        beta = float(stats.get("beta", 1.0))
        total = alpha + beta
        win_rate = alpha / total if total > 0 else 0.5
        n_obs = max(0.0, total - 2.0)  # exclude prior

        scores.append({
            "mid": str(mid),
            "win_rate": win_rate,
            "n_obs": n_obs,
            "total": total,
        })

    # ---- High IDs (champions) ----
    high_candidates = [s for s in scores if s["n_obs"] >= min_obs and s["win_rate"] >= high_win_rate_th]
    high_candidates.sort(key=lambda x: (-x["win_rate"], -x["n_obs"], x["mid"]))
    high_ids = [x["mid"] for x in high_candidates[:top_k]]

    # ---- Bad IDs (students) ----
    low_candidates = [s for s in scores if s["n_obs"] >= min_obs and s["win_rate"] <= low_win_rate_th]
    low_candidates.sort(key=lambda x: (x["win_rate"], -x["n_obs"], x["mid"]))
    bad_ids = [x["mid"] for x in low_candidates[:bottom_k]]

    # ---- Print ----
    print("📊 [Select Stats]")
    print(f"   - Total memories: {len(scores)}")
    print(f"   - High candidates: {len(high_candidates)} (top_k={top_k}, win>={high_win_rate_th}, obs>={min_obs})")
    print(f"   - Low candidates : {len(low_candidates)} (bottom_k={bottom_k}, win<={low_win_rate_th}, obs>={min_obs})")
    print(f"🔥 Output High IDs: {len(high_ids)}")
    print(f"🥶 Output Bad IDs : {len(bad_ids)}")

    return high_ids, bad_ids
