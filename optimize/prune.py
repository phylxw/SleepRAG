
def prune(memories, memory_stats):
    """
    执行高分记忆清理阶段 (Pruning)
    逻辑：在同一个聚类簇内，如果有表现非常好的记忆（状元），
    则清理掉那些表现非常差、或者是与状元差距过大的记忆。
    
    Args:
        memories (dict): 记忆库内容
        memory_stats (dict): 记忆的统计数据 (alpha, beta)
        
    Returns:
        set: 需要删除的 memory_ids 集合
    """
    print("\n========== 高分记忆清理阶段 (Pruning) ==========")
    to_delete_ids = set() 
    
    # 1. 构建聚类分组 map: cluster_id -> [mid, mid, ...]
    cluster_groups = {}
    for mid, rec in memories.items():
        cid = rec.get("cluster_id")
        if cid is not None:
            cid = int(cid)
            if cid not in cluster_groups: 
                cluster_groups[cid] = []
            cluster_groups[cid].append(mid)
    
    pruned_count = 0
    
    # 2. 遍历每个簇进行筛选
    for cid, members in cluster_groups.items():
        if len(members) < 2: continue # 只有一个人的簇不剪枝
        
        # 计算每个成员的分数统计
        member_stats_list = []
        for mid in members:
            # 获取统计信息，默认值为 alpha=1.0, beta=1.0 (Beta分布先验)
            stats = memory_stats.get(mid, {'alpha': 1.0, 'beta': 1.0})
            
            # 使用 .get 增加安全性
            alpha = stats.get('alpha', 1.0)
            beta = stats.get('beta', 1.0)
            total = alpha + beta
            
            # 计算胜率 (Win Rate)
            win_rate = alpha / total if total > 0 else 0.5
            
            member_stats_list.append({
                'id': mid, 
                'win_rate': win_rate, 
                'total': total
            })
            
        # 3. 排序：胜率高的排前面 (降序)，总数多的排前面 (降序)
        member_stats_list.sort(key=lambda x: (-x['win_rate'], -x['total']))
        
        # 拿到该簇里的“状元” (Best Memory)
        best_mem = member_stats_list[0]
        
        # 核心规则：只有当状元足够强 (胜率>0.7 且 尝试次数>4) 时，才敢动手删人
        if best_mem['win_rate'] > 0.7 and best_mem['total'] > 4:
            
            # 遍历剩下的“差生”
            for mem in member_stats_list[1:]:
                is_trash = False
                
                # 规则 A: 绝对垃圾 (胜率<0.3 且 尝试次数>4，确实扶不起来)
                if mem['win_rate'] < 0.3 and mem['total'] > 4: 
                    is_trash = True
                    
                # 规则 B: 相对垃圾 (状元太强 >=0.95，而你还没及格 <0.5，差距过大)
                if best_mem['win_rate'] >= 0.95 and mem['win_rate'] < 0.5: 
                    is_trash = True
                    
                if is_trash:
                    to_delete_ids.add(mem['id'])
                    pruned_count += 1
                    
    print(f"🗑️ [Pruning] 标记删除列表: {to_delete_ids}")
    print(f"✨ Pruning 完成，共清理: {pruned_count} 条")
    
    return to_delete_ids