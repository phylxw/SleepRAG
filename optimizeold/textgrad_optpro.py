import os
import re
import uuid
from omegaconf import DictConfig
# 假设这是你的工具库路径
from tools.optimize.callllm import init_llm, call_llm_batch
from tools.optimize.callexpert import init_expert_llm, call_expert, call_expert_batch
from tools.optimize.memoryload import load_clustered_memories, load_cluster_summary
# 保留原有导入
from optimize.prompt_generate import summarize_experience_prompt, expand_low_freq_memory_prompt
from utils.memorywrap import parse_memory

def textgrad_opt(cfg, memories, memory_stats, cluster_to_ids, bad_ids, to_delete_ids):
    """
    🔥 [Step 5] 执行 TextGrad 批量优化逻辑 (Expert Decision Guided)
    包含：梯度决策 (Expert Agent) -> 原语分发 (Refine/Expand/Replace) -> 执行优化 (Student)
    
    Args:
        cfg: Hydra 配置对象
        memories: 记忆库字典 (会被原地修改)
        memory_stats: 统计信息字典 (需要支持动态新增)
        cluster_to_ids: 聚类反向索引
        bad_ids: 待优化的低分 ID 列表
        to_delete_ids: 已经被标记删除的 ID 集合 (用于过滤)
        
    Returns:
        set: 本轮被成功优化的 ID 集合
    """
    print("\n========== TextGrad 记忆修正阶段 (Expert Batch Guided with Primitives) ==========")
    log_file_path = cfg.paths.get("lowfreq_textgrad_log", "textgrad_debug_log.txt")
    print(f"📝 调试日志将写入: {log_file_path}")
    # 1. 过滤出真正需要处理的 ID
    low_expand_ids = [mid for mid in bad_ids if mid in memories and mid not in to_delete_ids]
    print(f"🎯 待优化目标数量: {len(low_expand_ids)}")
    if low_expand_ids:
        print(f"   🆔 ID 列表: {low_expand_ids}")
    batch_size = cfg.optimizer.llm_batch_size
    optimized_ids = set()

    # 2. 按 Chunk 遍历处理
    for i in range(0, len(low_expand_ids), batch_size):
        chunk_ids = low_expand_ids[i : i + batch_size]
        
        # --- Step 5.1: 准备专家 Prompts (Diagnosis Phase) ---
        grad_prompts = []
        grad_metadata = [] 
        
        for mid in chunk_ids:
            rec = memories[mid]
            base_text = rec.get("contents", "")
            cluster_id = rec.get("cluster_id")
            stats = memory_stats.get(mid, {})
            neg_queries = stats.get('neg_queries', [])
            
            # --- 寻找优等生 (Momentum / Good Neighbors) ---
            good_neighbors_text = []
            if cluster_id is not None:
                cluster_id = int(cluster_id)
                members = cluster_to_ids.get(cluster_id, [])
                for m_id in members:
                    if str(m_id) == mid: continue
                    s = memory_stats.get(str(m_id), {})
                    s_total = s.get('alpha', 0) + s.get('beta', 0)
                    # 只有胜率 > 0.8 的才算好榜样
                    if s_total > 0 and (s.get('alpha', 0)/s_total) > 0.8:
                        good_neighbors_text.append(memories[str(m_id)].get("contents", ""))
                good_neighbors_text = good_neighbors_text[:3]
            good_examples_str = "\n".join([f"- {t}" for t in good_neighbors_text])
            
            # --- 构建专家输入 ---
            # 如果有负反馈（错题），则进入“专家决策模式”
            if len(neg_queries) > 0:
                # 获取 Top-K 错题
                top_k = cfg.optimizer.get("top_k_neg_queries", 3)
                selected_negs = neg_queries[:top_k]
                neg_text = "\n".join([f"- {q}" for q in selected_negs])
                
                # 🔥 直接使用 Config 中的决策 Prompt，而不是调用固定函数
                decision_prompt = cfg.optimizer.prompts.low_grad_expert.format(
                    content=base_text,
                    neg_queries=neg_text
                )
                
                grad_prompts.append(decision_prompt)
                grad_metadata.append({
                    "mid": mid, 
                    "need_expert": True,
                    "expert_prompt_content": decision_prompt,  # <--- 🔥 新增：存下专家Prompt
                    "base_text": base_text,
                    "neg_text": neg_text, # 存下来，REPLACE/EXPAND 要用
                    "good_examples_str": good_examples_str,
                    "good_neighbors_text": good_neighbors_text, 
                    "err_count": len(neg_queries)
                })
            else:
                # 没有错题，只有低分/低频 -> 进入降级模式 (Imitation/Reflection)
                grad_metadata.append({
                    "mid": mid, 
                    "need_expert": False,
                    "base_text": base_text,
                    "good_neighbors_text": good_neighbors_text,
                    "good_examples_str": good_examples_str,
                })

        # --- Step 5.2: 批量调用专家 (Expert Execution) ---
        expert_outputs = []
        if grad_prompts:
            print(f" 🧠 [Expert-Batch] 正在分析 {len(grad_prompts)} 条梯度并生成决策...")
            expert_outputs = call_expert_batch(grad_prompts, cfg)
        
    # --- Step 5.3: 解析决策并分发学生任务 (Dispatch Phase) ---
        student_prompts = []
        student_metadata = [] # 记录 task 类型和 ID
        expert_idx = 0 
        
        for meta in grad_metadata:
            mid = meta['mid']
            # 初始化日志对象
            log_info = {
                "mid": mid,
                "expert_prompt": meta.get("expert_prompt_content", "N/A"),
                "expert_output": "N/A",
                "action": "N/A",
                "gradient": "N/A",
                "student_prompt": ""
            }

            if meta['need_expert']:
                expert_resp = expert_outputs[expert_idx] if expert_idx < len(expert_outputs) else ""
                expert_idx += 1
                log_info["expert_output"] = expert_resp
                
                if expert_resp:
                    # 正则解析
                    action_match = re.search(r'\\box\{(REFINE|EXPAND|REPLACE)\}', expert_resp)
                    advice_match = re.search(r'\\advice\{(.*?)\}', expert_resp, re.DOTALL)
                    
                    action = action_match.group(1) if action_match else "REFINE" 
                    gradient = advice_match.group(1).strip() if advice_match else expert_resp
                    
                    log_info["action"] = action
                    log_info["gradient"] = gradient
                    
                    # === 原语分发逻辑 ===
                    
                    # 1. REFINE (优化)
                    if action == "REFINE":
                        reconstruct_tpl = cfg.optimizer.prompts.appgrad_low_refine
                        prompt = reconstruct_tpl.format(content=meta['base_text'], gradient=gradient)                       
                        log_info["student_prompt"] = prompt
                        student_prompts.append(prompt)
                        # 🔥 [修复点 1] 加上 "log": log_info
                        student_metadata.append({
                            "mid": mid, 
                            "type": "refine", 
                            "opt_type": "expert_refine", 
                            "log": log_info  # <--- 必须加这个！
                        })
                        
                    # 2. REPLACE (删增/替换)
                    elif action == "REPLACE":
                        reconstruct_tpl = cfg.optimizer.prompts.appgrad_low_replace
                        prompt = reconstruct_tpl.format(neg_queries=meta['neg_text'], gradient=gradient)
                        log_info["student_prompt"] = prompt
                        student_prompts.append(prompt)
                        # 🔥 [修复点 2] 加上 "log": log_info
                        student_metadata.append({
                            "mid": mid, 
                            "type": "replace", 
                            "opt_type": "expert_replace", 
                            "log": log_info 
                        })
                        
                    # 3. EXPAND (增加)
                    elif action == "EXPAND":
                        # Task A
                        reconstruct_tpl = cfg.optimizer.prompts.appgrad_low_refine
                        prompt_a = reconstruct_tpl.format(content=meta['base_text'], gradient=gradient)    
                        
                        log_info_a = log_info.copy()
                        log_info_a["student_prompt"] = prompt_a
                        log_info_a["action"] = "EXPAND (Part A: Refine Old)"
                        
                        student_prompts.append(prompt_a)
                        # 🔥 [修复点 3] 加上 "log": log_info_a
                        student_metadata.append({
                            "mid": mid, 
                            "type": "refine", 
                            "opt_type": "expert_expand_old", 
                            "log": log_info_a 
                        })
                        
                        # Task B
                        new_id = str(uuid.uuid4())
                        reconstruct_tpl = cfg.optimizer.prompts.appgrad_low_replace
                        prompt_b = reconstruct_tpl.format(neg_queries=meta['neg_text'], gradient=gradient)
                        
                        log_info_b = log_info.copy()
                        log_info_b["student_prompt"] = prompt_b
                        log_info_b["mid"] = new_id
                        log_info_b["action"] = "EXPAND (Part B: Create New)"

                        student_prompts.append(prompt_b)
                        # 🔥 [修复点 4] 加上 "log": log_info_b
                        student_metadata.append({
                            "mid": new_id, 
                            "type": "create", 
                            "opt_type": "expert_expand_new", 
                            "log": log_info_b 
                        })
                        
                else:
                    # 专家失败回退
                    if meta['good_neighbors_text']:
                        prompt = summarize_experience_prompt(meta['base_text'], meta['good_neighbors_text'], cfg)
                        log_info["action"] = "FALLBACK (Imitation)"
                        log_info["student_prompt"] = prompt
                        student_prompts.append(prompt)
                        # 🔥 [修复点 5] 加上 "log": log_info
                        student_metadata.append({
                            "mid": mid, 
                            "type": "refine", 
                            "opt_type": "neighbor_imitation", 
                            "log": log_info 
                        })
            
            else:
                # 非专家模式
                if meta['good_neighbors_text']:
                    prompt = summarize_experience_prompt(meta['base_text'], meta['good_neighbors_text'], cfg)
                    opt_type = "neighbor_imitation"
                    log_info["action"] = "IMITATION (No Expert)"
                else:
                    prompt = expand_low_freq_memory_prompt(meta['base_text'], "", cfg)
                    opt_type = "self_reflection"
                    log_info["action"] = "REFLECTION (No Expert)"
                
                log_info["student_prompt"] = prompt
                student_prompts.append(prompt)
                # 🔥 [修复点 6] 加上 "log": log_info
                student_metadata.append({
                    "mid": mid, 
                    "type": "refine", 
                    "opt_type": opt_type, 
                    "log": log_info 
                })

        # --- Step 5.4: 批量调用学生 (Student Execution) ---
        if student_prompts:
            print(f" 🚀 [Student-Batch] 正在执行 {len(student_prompts)} 项优化任务 (Refine/Replace/Create)...")
            # 这里建议用 call_llm_batch (学生模型)，如果你想用专家模型写也可以维持 call_expert_batch
            outputs = call_llm_batch(student_prompts, cfg) 

            # [修改点 4] 打开文件准备追加写入 (Append Mode)
            with open(log_file_path, "a", encoding="utf-8") as log_f:
                for meta, raw_output in zip(student_metadata, outputs):
                    output_text = parse_memory(raw_output)
                    # --- 🔥 写入日志的核心逻辑 ---
                    if "log" in meta:
                        info = meta["log"]
                        log_entry = (
                            f"\n{'='*40}\n"
                            f"🆔 Memory ID: {info['mid']} | Type: {meta['type']}\n"
                            f"--- 🧠 Expert Prompt ---\n{info['expert_prompt']}\n\n"
                            f"--- 🗣️ Expert Output ---\n{info['expert_output']}\n\n"
                            f"--- 📦 Parsed Action ---\nPrimitive: {info['action']}\nGradient: {info['gradient']}\n\n"
                            f"--- 📝 Student Prompt ---\n{info['student_prompt']}\n\n"
                            f"--- ✨ New Memory Content ---\n{output_text}\n"
                            f"{'='*40}\n"
                        )
                        log_f.write(log_entry)
                        log_f.flush() # 强制刷新缓冲区，防止程序崩溃丢失日志
            
                    if output_text and len(output_text) > 10:
                        target_mid = meta['mid']
                        task_type = meta['type']
                        
                        if task_type in ["refine", "replace"]:
                            # 原地更新 (Update)
                            if target_mid in memories:
                                memories[target_mid]["contents"] = output_text
                                memories[target_mid]["opt_type"] = meta['opt_type']
                                optimized_ids.add(target_mid)
                                
                        elif task_type == "create":
                            # 新增插入 (Insert)
                            print(f"  ✨ [EXPAND] 正在分裂产生新记忆 ID: {target_mid[:8]}...")
                            memories[target_mid] = {
                                "id": target_mid,
                                "contents": output_text,
                                "cluster_id": -1, # 新记忆暂时游离，等待下一轮聚类分配
                                "opt_type": meta['opt_type']
                            }
                            # 重要：初始化统计信息，避免下一轮报错
                            memory_stats[target_mid] = {
                                "alpha": 1.0, 
                                "beta": 1.0, 
                                "neg_queries": []
                            }
                            optimized_ids.add(target_mid)

    return optimized_ids