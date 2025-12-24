import os
import json
import time
from typing import Dict, List, Tuple, Set

def load_clustered_memories(path: str) -> Tuple[Dict[str, dict], List[str]]:
    memories: Dict[str, dict] = {}
    order: List[str] = []
    print(f"📥 正在加载聚类后的记忆文件: {path}")
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return {}, []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            mid = str(obj["id"])
            memories[mid] = obj
            order.append(mid)
    print(f"✅ 共加载 {len(memories)} 条记忆")
    return memories, order


def load_cluster_summary(path: str) -> Dict[int, List[str]]:
    cluster_to_ids: Dict[int, List[str]] = {}
    print(f"📥 正在加载聚类摘要文件: {path}")
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            cid = int(obj["cluster_id"])
            ids = [str(x) for x in obj.get("memory_ids", [])]
            cluster_to_ids[cid] = ids
    print(f"✅ 共加载 {len(cluster_to_ids)} 个聚类")
    return cluster_to_ids


def load_memory_freq(path: str) -> Dict[str, int]:
    freq_map: Dict[str, int] = {}
    print(f"📥 正在加载记忆频次文件: {path}")
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            # 兼容 memory_id 或 id 字段
            mid = str(obj.get("memory_id", obj.get("id", "")))
            if not mid: continue
            freq = int(obj.get("freq", 0))
            freq_map[mid] = freq
    print(f"✅ 频次记录数: {len(freq_map)}")
    return freq_map