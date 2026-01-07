import os
import re
import uuid
import logging
from typing import Set, List, Dict
from omegaconf import DictConfig

# 工具库导入
from tools.optimize.callllm import call_llm_batch
from tools.optimize.callllm import init_llm  # 如果需要重新初始化
from tools.optimize.callexpert import call_expert_batch
from utils.memorywrap import parse_memory


def evolve_high_score_opt(cfg: DictConfig, memories: Dict, memory_stats: Dict, high_ids: List[str]) -> Set[str]:
    """
    🏆 [Step 4.5] 高分记忆进化阶段 (Ace Evolution)
    
    策略：
        1. 筛选有错题的高分记忆。
        2. 专家诊断：IGNORE (忽略), SUPPLEMENT (补充), SPLIT (分裂)。
        3. 学生执行：生成新的记忆内容。
        4. 写入日志与记忆库。
    
    Args:
        cfg: Hydra配置
        memories: 记忆库 (In-place modification)
        memory_stats: 统计信息 (In-place modification)
        high_ids: 高分记忆 ID 列表
        
    Returns:
        Set[str]: 新生成的记忆 ID 集合
    """
    print("\n========== 高分记忆进化阶段 (Ace Evolution) ==========")
    
    # --- 1. 环境与日志准备 ---
    # 强制指定日志路径
    log_file_path = cfg.paths.get("highfreq_textgrad_log", "textgrad_debug_log.txt")
    
    # 确保目录存在
    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    except Exception as e:
        print(f"⚠️ 无法创建日志目录: {e}")

    print(f"📝 进化日志将追加至: {log_file_path}")

    # --- 2. 目标筛选 (Target Selection) ---
    target_ids = []
    for mid in high_ids:
        # 防御性检查：确保ID存在且有统计数据
        if mid not in memories: continue
        stats = memory_stats.get(mid, {})
        neg_queries = stats.get('neg_queries', [])
        
        # 只有当存在错题时，才需要进化
        if len(neg_queries) > 0:
            target_ids.append(mid)

    # 调试截断逻辑 (Debug Limit)
    max_count = cfg.optimizer.get("max_high_opt_count", 5)
    if len(target_ids) > max_count:
        print(f"✂️ [Evolve] 命中高分优化上限: {len(target_ids)} -> {max_count}")
        target_ids = target_ids[:max_count]
    
    print(f"💎 待进化的王牌记忆数量: {len(target_ids)}")
    if target_ids:
        print(f"   🆔 ID 列表: {target_ids}")
    if not target_ids:
        return set()

    # --- 3. 准备专家 Prompt (Expert Phase) ---
    expert_prompts = []
    evolve_metadata = []
    
    # 批处理大小控制
    batch_size = cfg.optimizer.llm_batch_size
    new_created_ids_total = set()

    for i in range(0, len(target_ids), batch_size):
        chunk_ids = target_ids[i : i + batch_size]
        expert_prompts = []
        chunk_metadata = []

        print(f" 🧠 [Expert-Batch] 正在处理第 {i} - {i+len(chunk_ids)} 条高分记忆...")

        for mid in chunk_ids:
            rec = memories[mid]
            base_text = rec.get("contents", "")
            stats = memory_stats.get(mid, {})
            neg_queries = stats.get('neg_queries', [])
            
            # 取前 K 个错题，避免 Context Window 爆炸
            top_k_neg = 5
            neg_text = "\n".join([f"- {q}" for q in neg_queries[:top_k_neg]])
            
            # 构造 Prompt
            try:
                prompt = cfg.optimizer.prompts.high_grad_expert.format(
                    content=base_text,
                    neg_queries=neg_text
                )
            except Exception as e:
                print(f"❌ Prompt 格式化失败 (MID: {mid}): {e}")
                continue

            expert_prompts.append(prompt)
            chunk_metadata.append({
                "mid": mid,
                "base_text": base_text,
                "neg_text": neg_text,
                "expert_prompt_content": prompt
            })

        if not expert_prompts:
            continue

        # 调用专家模型
        expert_outputs = call_expert_batch(expert_prompts, cfg)

        # --- 4. 解析决策并分发 (Dispatch Phase) ---
        student_prompts = []
        student_tasks = [] # 存储待执行的任务信息

        for meta, expert_resp in zip(chunk_metadata, expert_outputs):
            mid = meta['mid']
            
            # 初始化日志对象
            log_info = {
                "mid": mid,
                "type": "evolve_high_score",
                "expert_prompt": meta["expert_prompt_content"],
                "expert_output": expert_resp,
                "action": "UNKNOWN",
                "gradient": "N/A",
                "split_num": 0,
                "student_prompt": "N/A"
            }

            if not expert_resp:
                print(f"⚠️ 专家模型返回为空 (MID: {mid})")
                continue

            # === 正则解析 ===
            # 1. 提取 Action: \box{IGNORE} / \box{SUPPLEMENT} / \box{SPLIT}
            action_match = re.search(r'\\box\{(IGNORE|SUPPLEMENT|SPLIT)\}', expert_resp)
            action = action_match.group(1) if action_match else "IGNORE" # 默认保守策略：忽略
            
            # 2. 提取 Gradient (建议): \gradient{...}
            # 使用 DOTALL 匹配跨行文本
            gradient_match = re.search(r'\\gradient\{(.*?)\}', expert_resp, re.DOTALL)
            gradient = gradient_match.group(1).strip() if gradient_match else "No specific advice provided."
            
            # 3. 提取 Num (仅 SPLIT): \num{...}
            num_match = re.search(r'\\num\{(\d+)\}', expert_resp)
            split_num = int(num_match.group(1)) if num_match else 1

            # 更新日志信息
            log_info["action"] = action
            log_info["gradient"] = gradient
            if action == "SPLIT":
                log_info["split_num"] = split_num

            # === 任务分发 ===
            if action == "IGNORE":
                # 直接记录日志，不调用学生
                _write_log(log_file_path, log_info, "Skipped (IGNORE Action)")
                continue

            elif action == "SUPPLEMENT":
                # 生成单条补充记忆
                reconstruct_tpl = cfg.optimizer.prompts.appgrad_high_supplement
                s_prompt = reconstruct_tpl.format(content=meta['base_text'], gradient=gradient)
                log_info["student_prompt"] = s_prompt
                
                student_prompts.append(s_prompt)
                student_tasks.append({
                    "parent_mid": mid,
                    "action": "SUPPLEMENT",
                    "log": log_info
                })

            elif action == "SPLIT":
                reconstruct_tpl = cfg.optimizer.prompts.appgrad_high_split
                s_prompt = reconstruct_tpl.format(neg_text=meta['neg_text'], gradient=gradient)
                log_info["student_prompt"] = s_prompt
                student_prompts.append(s_prompt)
                student_tasks.append({
                    "parent_mid": mid,
                    "action": "SPLIT",
                    "log": log_info
                })

        # --- 5. 学生执行与结果保存 (Student Phase) ---
        if student_prompts:
            print(f" 🚀 [Student-Batch] 正在执行 {len(student_prompts)} 项进化任务...")
            student_outputs = call_llm_batch(student_prompts, cfg)

            for task, raw_output in zip(student_tasks, student_outputs):
                output_text = parse_memory(raw_output)
                log_info = task["log"]
                parent_mid = task["parent_mid"]
                action_type = task["action"]
                
                # 结果处理容器
                final_results_for_log = [] 

                if action_type == "SUPPLEMENT":
                    if output_text and len(output_text) > 10:
                        new_id = str(uuid.uuid4())
                        _save_new_memory(memories, memory_stats, new_id, output_text, parent_mid, "high_score_supplement")
                        new_created_ids_total.add(new_id)
                        final_results_for_log.append(f"[ID: {new_id}] {output_text[:100]}...")
                        print(f"  ✨ [SUPPLEMENT] 为 {parent_mid[:8]} 增加副官: {new_id[:8]}")
                    else:
                        final_results_for_log.append("FAILED: Output too short.")

                elif action_type == "SPLIT":
                    # 按照分隔符切分
                    delimiter = "==========SPLIT=========="
                    raw_splits = output_text.split(delimiter)
                    # 过滤空字符串
                    valid_splits = [s.strip() for s in raw_splits if len(s.strip()) > 10]
                    
                    if valid_splits:
                        print(f"  ✨ [SPLIT] 记忆 {parent_mid[:8]} 分裂出 {len(valid_splits)} 条新知识")
                        for idx, content in enumerate(valid_splits):
                            new_id = str(uuid.uuid4())
                            _save_new_memory(memories, memory_stats, new_id, content, parent_mid, f"high_score_split_{idx+1}")
                            new_created_ids_total.add(new_id)
                            final_results_for_log.append(f"[ID: {new_id}] {content[:100]}...")
                    else:
                        final_results_for_log.append("FAILED: No valid splits found.")

                # --- 6. 写入日志 (Write Log) ---
                # 将最终生成的记忆内容摘要合并写入日志
                result_summary = "\n".join(final_results_for_log)
                _write_log(log_file_path, log_info, result_summary)

    print(f"✅ [Evolve] 进化完成，共新增 {len(new_created_ids_total)} 条高阶记忆")
    return new_created_ids_total


# ------------------------------------------------------------------------------
# 内部私有辅助函数 (保持主逻辑简洁)
# ------------------------------------------------------------------------------

def _save_new_memory(memories, memory_stats, new_id, content, parent_id, opt_type):
    """
    将新生成的记忆安全地保存到字典中，并初始化统计信息。
    """
    # 1. 保存到记忆库
    memories[new_id] = {
        "id": new_id,  # 🔥 核心：必须包含 ID 字段
        "contents": content,
        "cluster_id": -1, # 标记为未聚类，等待下一轮处理
        "opt_type": opt_type,
        "parent_id": parent_id # 记录血缘关系（可用于后续追溯）
    }
    
    # 2. 初始化统计 (给一个公平的初始分，比如 alpha=1, beta=1)
    memory_stats[new_id] = {
        "alpha": 1.0,
        "beta": 1.0,
        "neg_queries": [],
        "pos_queries": []
    }

def _write_log(file_path, info, result_content):
    """
    将单条处理记录追加写入到 TXT 文件。
    """
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            log_entry = (
                f"\n{'='*60}\n"
                f"🕒 Processing Log: Ace Evolution\n"
                f"🆔 Parent Memory ID: {info['mid']}\n"
                f"--- 🧠 Expert Prompt (Input) ---\n{info['expert_prompt']}\n\n"
                f"--- 🗣️ Expert Output (Raw) ---\n{info['expert_output']}\n\n"
                f"--- 📦 Parsed Decision ---\n"
                f"   Action   : {info['action']}\n"
                f"   Gradient : {info['gradient']}\n"
                f"   Split Num: {info.get('split_num', 0)}\n\n"
                f"--- 📝 Student Prompt ---\n{info['student_prompt']}\n\n"
                f"--- ✨ Final Result (New Memories) ---\n{result_content}\n"
                f"{'='*60}\n"
            )
            f.write(log_entry)
            f.flush()
    except Exception as e:
        print(f"⚠️ 日志写入异常: {e}")