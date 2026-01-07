from __future__ import annotations
import math
from typing import Dict, Any, Optional, Set

try:
    from omegaconf import DictConfig
except Exception:  # pragma: no cover
    DictConfig = Any  # type: ignore


def _beta_mean(alpha: float, beta: float) -> float:
    s = alpha + beta
    return alpha / s if s > 0 else 0.5


def _beta_var(alpha: float, beta: float) -> float:
    # Var(Beta(a,b)) = ab / ((a+b)^2 (a+b+1))
    s = alpha + beta
    if s <= 1:
        return 0.25
    return (alpha * beta) / (s * s * (s + 1.0))


def _lcb(alpha: float, beta: float, z: float) -> float:
    mu = _beta_mean(alpha, beta)
    var = _beta_var(alpha, beta)
    return mu - z * math.sqrt(max(var, 1e-12))


def _ucb(alpha: float, beta: float, z: float) -> float:
    mu = _beta_mean(alpha, beta)
    var = _beta_var(alpha, beta)
    return mu + z * math.sqrt(max(var, 1e-12))


def _safe_get_cfg(cfg: Any, path: str, default: Any) -> Any:
    """Access nested config with both DictConfig and dict-like objects."""
    if cfg is None:
        return default
    cur = cfg
    for key in path.split("."):
        try:
            if isinstance(cur, dict):
                cur = cur.get(key, None)
            else:
                cur = getattr(cur, key) if hasattr(cur, key) else cur.get(key)  # type: ignore
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def prune(memories: Dict[str, Dict[str, Any]], memory_stats: Dict[str, Dict[str, Any]], cfg: Optional[Any] = None) -> Set[str]:
    """
    Safe, confidence-based pruning within clusters.

    Each memory i has posterior p_i ~ Beta(alpha_i, beta_i). In a cluster, if the
    best memory is strongly good (LCB >= threshold with enough observations),
    we delete only those memories that are strongly bad (UCB <= threshold with
    enough observations). This avoids deleting low-data items where uncertainty
    is high.

    Config (optional, via cfg):
      optimizer.prune.enabled: bool (default True)
      optimizer.prune.verbose: bool (default True)
      optimizer.prune.z: float (default 1.64)  # ~90% one-sided
      optimizer.prune.min_cluster_size: int (default 2)
      optimizer.prune.skip_negative_cluster: bool (default True)  # skip cluster_id < 0
      optimizer.prune.min_obs_best: int (default 6)
      optimizer.prune.best_lcb_thr: float (default 0.70)
      optimizer.prune.trash_ucb_thr: float (default 0.35)
      optimizer.prune.max_prune_per_cluster: int (default 3)
    """
    enabled = bool(_safe_get_cfg(cfg, "optimizer.prune.enabled", True))
    verbose = bool(_safe_get_cfg(cfg, "optimizer.prune.verbose", True))
    if not enabled:
        if verbose:
            print("\n========== Pruning ==========")
            print("ℹ️ [Pruning] Disabled by config; skipping.")
        return set()

    z = float(_safe_get_cfg(cfg, "optimizer.prune.z", 1.64))
    min_cluster_size = int(_safe_get_cfg(cfg, "optimizer.prune.min_cluster_size", 2))
    skip_negative_cluster = bool(_safe_get_cfg(cfg, "optimizer.prune.skip_negative_cluster", True))

    min_obs_best = int(_safe_get_cfg(cfg, "optimizer.prune.min_obs_best", 6))
    best_lcb_thr = float(_safe_get_cfg(cfg, "optimizer.prune.best_lcb_thr", 0.70))
    trash_ucb_thr = float(_safe_get_cfg(cfg, "optimizer.prune.trash_ucb_thr", 0.35))
    max_prune_per_cluster = int(_safe_get_cfg(cfg, "optimizer.prune.max_prune_per_cluster", 3))

    if verbose:
        print("\n========== 高分记忆清理阶段 (Pruning, Safe) ==========")
        print(
            f"Config: z={z}, min_obs_best={min_obs_best}, best_lcb_thr={best_lcb_thr}, "
            f"trash_ucb_thr={trash_ucb_thr}, max_prune_per_cluster={max_prune_per_cluster}"
        )

    # cluster_id -> [mid, ...]
    cluster_groups: Dict[int, list[str]] = {}
    for mid, rec in memories.items():
        cid = rec.get("cluster_id", None)
        if cid is None:
            continue
        try:
            cid_int = int(cid)
        except Exception:
            continue
        if skip_negative_cluster and cid_int < 0:
            continue
        cluster_groups.setdefault(cid_int, []).append(str(mid))

    to_delete: Set[str] = set()
    pruned_count = 0
    considered_clusters = 0

    for cid, members in cluster_groups.items():
        if len(members) < min_cluster_size:
            continue
        considered_clusters += 1

        stats_list = []
        for mid in members:
            st = memory_stats.get(mid, None)
            if st is None:
                try:
                    st = memory_stats.get(int(mid), None)  # type: ignore[arg-type]
                except Exception:
                    st = None
            if st is None:
                st = {"alpha": 1.0, "beta": 1.0}

            alpha = float(st.get("alpha", 1.0))
            beta = float(st.get("beta", 1.0))

            # Using Beta(1,1) prior -> true observations = (a+b-2)
            n_obs = max(0.0, alpha + beta - 2.0)

            stats_list.append(
                {
                    "id": str(mid),
                    "alpha": alpha,
                    "beta": beta,
                    "n_obs": n_obs,
                    "mean": _beta_mean(alpha, beta),
                    "lcb": _lcb(alpha, beta, z=z),
                    "ucb": _ucb(alpha, beta, z=z),
                }
            )

        # deterministic best: mean desc, n_obs desc, id asc
        stats_list.sort(key=lambda x: (-x["mean"], -x["n_obs"], x["id"]))
        best = stats_list[0]

        # prune only when the best is truly strong
        if best["n_obs"] < min_obs_best or best["lcb"] < best_lcb_thr:
            continue

        pruned_here = 0
        for mem in stats_list[1:]:
            if pruned_here >= max_prune_per_cluster:
                break

            # never delete under-observed items (uncertainty high)
            if mem["n_obs"] < min_obs_best:
                continue

            # high-confidence bad: even optimistic estimate is low
            if mem["ucb"] < trash_ucb_thr:
                to_delete.add(mem["id"])
                pruned_here += 1
                pruned_count += 1

        if verbose and pruned_here > 0:
            print(
                f"[Cluster {cid}] best(id={best['id']}, mean={best['mean']:.3f}, lcb={best['lcb']:.3f}, n={best['n_obs']:.0f}) "
                f"=> pruned {pruned_here}"
            )

    if verbose:
        print(f"🗑️ [Pruning] 标记删除列表: {to_delete}")
        print(f"✨ Pruning 完成，共清理: {pruned_count} 条 (clusters considered: {considered_clusters})")

    return to_delete
