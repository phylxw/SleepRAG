# Hydra
import hydra
from omegaconf import DictConfig
from tools.optimize.callllm import init_llm, call_llm_batch
from tools.optimize.callexpert import init_expert_llm, call_expert,call_expert_batch
from tools.optimize.memoryload import load_clustered_memories, load_cluster_summary
from optimize.prompt_generate import generate_gradient_prompt,apply_gradient_prompt,summarize_experience_prompt,expand_low_freq_memory_prompt

def textgrad_opt(cfg, memories, memory_stats, cluster_to_ids, bad_ids, to_delete_ids):
    """
    🔥 [Step 5] 执行 TextGrad 批量优化逻辑
    包含：梯度计算 (Expert) -> 梯度应用/模仿/反思 (Student)
    
    Args:
        cfg: Hydra 配置对象
        memories: 记忆库字典 (会被原地修改)
        memory_stats: 统计信息字典
        cluster_to_ids: 聚类反向索引
        bad_ids: 待优化的低分 ID 列表
        to_delete_ids: 已经被标记删除的 ID 集合 (用于过滤)
        
    Returns:
        set: 本轮被成功优化的 ID 集合
    """
    print("\n========== TextGrad 记忆修正阶段 (Expert Batch Guided) ==========")
    
    # 1. 过滤出真正需要处理的 ID
    low_expand_ids = [mid for mid in bad_ids if mid in memories and mid not in to_delete_ids]
    print(f"🎯 待优化目标数量: {len(low_expand_ids)}")

    batch_size = cfg.optimizer.llm_batch_size
    optimized_ids = set()

    # 2. 按 Chunk 遍历处理
    for i in range(0, len(low_expand_ids), batch_size):
        chunk_ids = low_expand_ids[i : i + batch_size]
        
        # --- Step 5.1: 准备专家 Prompts ---
        grad_prompts = []
        grad_metadata = [] 
        
        for mid in chunk_ids:
            rec = memories[mid]
            base_text = rec.get("contents", "")
            cluster_id = rec.get("cluster_id")
            stats = memory_stats.get(mid, {})
            neg_queries = stats.get('neg_queries', [])
            
            # 找优等生 (Momentum)
            good_neighbors_text = []
            if cluster_id is not None:
                cluster_id = int(cluster_id)
                members = cluster_to_ids.get(cluster_id, [])
                for m_id in members:
                    if str(m_id) == mid: continue
                    s = memory_stats.get(str(m_id), {})
                    s_total = s.get('alpha', 0) + s.get('beta', 0)
                    if s_total > 0 and (s.get('alpha', 0)/s_total) > 0.8:
                        good_neighbors_text.append(memories[str(m_id)].get("contents", ""))
                good_neighbors_text = good_neighbors_text[:3]
            good_examples_str = "\n".join([f"- {t}" for t in good_neighbors_text])
            
            # 判断是否需要专家介入
            if len(neg_queries) > 0:
                prompt = generate_gradient_prompt(base_text, neg_queries, cfg) # 注意这里我加了 cfg，原函数如果需要的话
                grad_prompts.append(prompt)
                grad_metadata.append({
                    "mid": mid, 
                    "need_expert": True,
                    "base_text": base_text,
                    "good_examples_str": good_examples_str,
                    "good_neighbors_text": good_neighbors_text, 
                    "err_count": len(neg_queries)
                })
            else:
                grad_metadata.append({
                    "mid": mid, 
                    "need_expert": False,
                    "base_text": base_text,
                    "good_neighbors_text": good_neighbors_text,
                    "good_examples_str": good_examples_str,
                })

        # --- Step 5.2: 批量调用专家 ---
        gradients = []
        if grad_prompts:
            print(f" 🧠 [Expert-Batch] 正在分析 {len(grad_prompts)} 条梯度...")
            # 注意：确保 call_expert_batch 在这个作用域可用
            gradients = call_expert_batch(grad_prompts, cfg)
        
        # --- Step 5.3: 准备学生 Prompts ---
        student_prompts = []
        student_metadata = []
        grad_idx = 0 
        
        for meta in grad_metadata:
            mid = meta['mid']
            opt_type = "unknown"
            prompt = ""
            
            if meta['need_expert']:
                gradient_text = gradients[grad_idx] if grad_idx < len(gradients) else None
                grad_idx += 1
                
                if gradient_text:
                    prompt = apply_gradient_prompt(meta['base_text'], gradient_text, meta['good_examples_str'], cfg)
                    opt_type = f"textgrad_{meta['err_count']}_errors"
                else:
                    # 降级策略
                    if meta['good_neighbors_text']:
                        prompt = summarize_experience_prompt(meta['base_text'], meta['good_neighbors_text'], cfg)
                        opt_type = "neighbor_imitation"
                    else:
                        prompt = expand_low_freq_memory_prompt(meta['base_text'], "", cfg)
                        opt_type = "self_reflection"
            else:
                if meta['good_neighbors_text']:
                    prompt = summarize_experience_prompt(meta['base_text'], meta['good_neighbors_text'], cfg)
                    opt_type = "neighbor_imitation"
                else:
                    prompt = expand_low_freq_memory_prompt(meta['base_text'], "", cfg)
                    opt_type = "self_reflection"
            
            student_prompts.append(prompt)
            student_metadata.append({"mid": mid, "opt_type": opt_type})

        # --- Step 5.4: 批量调用学生 ---
        if student_prompts:
            print(f" 🚀 [Student-Batch] 正在优化 {len(student_prompts)} 条记忆...")
            # 假设你用同一个函数调用 LLM，或者这里应该是 call_student_batch
            outputs = call_expert_batch(student_prompts, cfg) 
            
            for meta, output_text in zip(student_metadata, outputs):
                if output_text and len(output_text) > 10:
                    mid = meta['mid']
                    # 原地修改 memories
                    memories[mid]["contents"] = output_text
                    memories[mid]["opt_type"] = meta['opt_type']
                    optimized_ids.add(mid)
                    
    return optimized_ids