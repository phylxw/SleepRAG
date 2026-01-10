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

_MEMORY_BLOCK_RE = re.compile(r"\\memory\{(.*)\}", re.DOTALL)

def _extract_single_memory(raw_output: str) -> str:
    if not raw_output:
        return ""
    # Primary: project parser
    txt = (parse_memory(raw_output) or "").strip()
    if txt:
        return txt
    # Fallback: regex
    m = _MEMORY_BLOCK_RE.search(raw_output)
    return (m.group(1).strip() if m else "")

def _basic_guard(text: str, *, min_len: int = 20, max_len: int = 2000) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < min_len or len(t) > max_len:
        return False
    banned = [
        "As an AI",
        "As a language model",
        "I can't",
        "I cannot",
        "I am unable",
        "抱歉",
        "无法",
    ]
    if any(b in t for b in banned):
        return False
    return True


# ------------------------------------------------------------------------------
# Acceptance test + rollback (retry) helpers
# ------------------------------------------------------------------------------
# Motivation:
# - Prevent "writeback pollution": do not overwrite the memory store unless the edit
#   is judged to improve the previously failed queries.
# - If a candidate fails the acceptance test, retry generation up to a small budget.
#
# Default behavior (if cfg is missing): enabled=True, max_retries=2.

_ACCEPTANCE_PROMPT = r'''
You are a Cognitive Logic Auditor for a RAG memory store.
[Failed Queries]
{failed_queries}

[Old Memory]
{old_memory}

[New Memory]
{new_memory}

[Audit Criteria]
1. **Methodology Check**: Does the New Memory explain the *reasoning logic*, *step-by-step derivation*, or *general principle*? (Reject if it just gives the factual answer).
2. **Generalization**: Is the logic abstract enough to apply to similar problems, not just the specific failed queries?
3. **Accuracy**: No hallucinations or uncertain facts.
4. **Atomicity**: Focuses on one core concept/framework.

[Output Format — STRICT]
Verdict: PASS|FAIL
Feedback: <If FAIL, explain specifically which logic is missing. If PASS, write "OK".>
'''

_VERDICT_RE = re.compile(r"Verdict:\s*(PASS|FAIL)", re.IGNORECASE)
_FEEDBACK_RE = re.compile(r"Feedback:\s*(.*)", re.IGNORECASE | re.DOTALL)

def _get_acceptance_params(cfg):
    """
    修改后：直接读取 cfg.parameters.acceptance 下的配置
    """
    max_retries = cfg.parameters.max_retries
    print(f'限制轮次是：{max_retries}')

    return True, max_retries

def _parse_acceptance(output: str):
    if not output:
        return {"verdict": "FAIL", "feedback": "No judge output."}
    m = _VERDICT_RE.search(output)
    verdict = (m.group(1).upper() if m else "FAIL")
    m2 = _FEEDBACK_RE.search(output)
    feedback = (m2.group(1).strip() if m2 else "").strip()
    if not feedback:
        feedback = "OK" if verdict == "PASS" else "Missing feedback."
    return {"verdict": verdict, "feedback": feedback}

def _acceptance_test_batch(cfg, items):
    prompts = []
    for it in items:
        prompts.append(_ACCEPTANCE_PROMPT.format(
            failed_queries=(it.get("failed_queries","") or "").strip(),
            old_memory=(it.get("old_memory","") or "").strip(),
            new_memory=(it.get("new_memory","") or "").strip(),
        ))
    if not prompts:
        return []
    judge_outs = call_expert_batch(prompts, cfg)
    return [_parse_acceptance(o) for o in judge_outs]

def _build_retry_prompt(original_student_prompt: str, prev_memory: str, judge_feedback: str) -> str:
    return (
        original_student_prompt
        + "\n\n[Previous Attempt]\n"
        + f"\\memory{{{(prev_memory or '').strip()}}}\n\n"
        + "[Judge Feedback]\n"
        + (judge_feedback or "").strip()
        + "\n\nRewrite again. Output ONLY the memory wrapper."
    )

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
    
    def tee_print(msg):
        """同时打印到终端并追加写入日志文件"""
        # print(msg) # 打印到屏幕
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n") # 写入文件
        except Exception:
            pass
    
    # 1. 过滤出真正需要处理的 ID
    bad_ids_list = list(bad_ids)
    if not isinstance(bad_ids, list):
        bad_ids_list.sort()
    low_expand_ids = [mid for mid in bad_ids_list if mid in memories and mid not in to_delete_ids]
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
                        
                        m_key = str(m_id)
                        if m_key not in memories:
                            continue
                        good_neighbors_text.append(memories[m_key].get("contents", ""))
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
                    
                    # ================= 打印代码 =================
                    tee_print(f"\n[Expert Logic] MID: {mid}")
                    tee_print(f"   >>> 🛠️ 原语 (Action): {action}")
                    tee_print(f"   >>> 🧠 梯度 (Gradient): {gradient[:20]}...{gradient[-20:]}") # 只打印40字
                    # ====================================================

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
                            "log": log_info,
                            "old_content": meta.get("base_text",""),
                            "neg_text": meta.get("neg_text",""),
                            "student_prompt": prompt,
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
                            "log": log_info,
                            "old_content": meta.get("base_text",""),
                            "neg_text": meta.get("neg_text",""),
                            "student_prompt": prompt,
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
                            "log": log_info_a,
                            "old_content": meta.get("base_text",""),
                            "neg_text": meta.get("neg_text",""),
                            "student_prompt": prompt_a,
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
                            "log": log_info_b,
                            "old_content": "",
                            "neg_text": meta.get("neg_text",""),
                            "student_prompt": prompt_b,
                            "parent_mid": mid,
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
            
            # ========== 🔥【新增打印 1】查看学生领到的任务单 ==========
            tee_print(f"   📋 学生任务详情:")
            for meta in student_metadata:
                # 获取原语类型 (REFINE/REPLACE/EXPAND)
                action_type = meta.get("type", "UNKNOWN").upper()
                # 获取更具体的策略 (如 expert_refine, neighbor_imitation)
                strategy = meta.get("opt_type", "N/A")
                mid_display = meta['mid'][:8] # 只显示ID前8位方便阅读
                
                tee_print(f"   -> 🆔 [{mid_display}] | 动作: {action_type} | 策略: {strategy}")
                # 如果你想看具体的 prompt 开头，可以把下面这行注释打开
                tee_print(f"      Prompt预览: {meta.get('student_prompt', '')[:60]}...")
            tee_print("   ------------------------------------------------")
            # =======================================================

            # 这里建议用 call_llm_batch (学生模型)，如果你想用专家模型写也可以维持 call_expert_batch
            outputs = call_llm_batch(student_prompts, cfg)

            # ========== 🔥【新增打印 2】查看学生初稿 ==========
            tee_print(f"\n   📨 [Student Output] 收到 {len(outputs)} 条初稿:")
            for meta, raw_txt in zip(student_metadata, outputs):
                # 稍微清洗一下换行符以便在一行显示
                clean_preview = raw_txt.strip().replace('\n', ' ')
                mid = meta['mid'][:8]
                tee_print(f"      -> 🆔 {mid} | 长度: {len(raw_txt)} | 预览: {clean_preview[0:20]}...{clean_preview[-20:]}")
            # =================================================

            min_len = int(getattr(cfg.optimizer, "min_memory_len", 20) or 20)
            max_len = int(getattr(cfg.optimizer, "max_memory_len", 2000) or 2000)

            candidates = []
            for meta, raw_output in zip(student_metadata, outputs):
                out_text = _extract_single_memory(raw_output)
                # out_text = raw_output
                candidates.append({
                    "meta": meta,
                    "attempts": [{"out_text": out_text, "judge": {}}],
                })

            acc_enabled, max_retries = _get_acceptance_params(cfg)

            def _need_accept(meta: dict) -> bool:
                return bool(acc_enabled and meta.get("neg_text"))

            # Judge attempt-0 (batched)
            judge_items, judge_map = [], []
            for i, cand in enumerate(candidates):
                meta = cand["meta"]
                out_text = cand["attempts"][-1]["out_text"]
                if not out_text or not _basic_guard(out_text, min_len=min_len, max_len=max_len):
                    continue
                if _need_accept(meta):
                    judge_items.append({
                        "failed_queries": meta.get("neg_text",""),
                        "old_memory": meta.get("old_content",""),
                        "new_memory": out_text,
                    })
                    judge_map.append(i)

            judge_results = _acceptance_test_batch(cfg, judge_items)
            for idx, res in zip(judge_map, judge_results):
                candidates[idx]["attempts"][-1]["judge"] = res

            # Retry loop (rollback)
            for _retry_idx in range(max_retries):
                retry_prompts, retry_indices = [], []
                for i, cand in enumerate(candidates):
                    meta = cand["meta"]
                    if not _need_accept(meta):
                        continue

                    last = cand["attempts"][-1]
                    out_text = (last.get("out_text") or "").strip()
                    judge = last.get("judge") or {}

                    ok_guard = bool(out_text and _basic_guard(out_text, min_len=min_len, max_len=max_len))
                    ok_accept = (judge.get("verdict") == "PASS")

                    if ok_guard and ok_accept:
                        continue

                    if not ok_guard:
                        feedback = (
                            "Rejected by basic guard. Make it concise, atomic, and factual. "
                            "Add a 'Keywords:' line with retrieval terms."
                        )
                    else:
                        feedback = judge.get("feedback") or "Still missing key info; be more specific and include retrieval keywords."

                    orig_prompt = meta.get("student_prompt") or (meta.get("log", {}) or {}).get("student_prompt", "")
                    if not orig_prompt:
                        continue

                    retry_prompts.append(_build_retry_prompt(orig_prompt, out_text, feedback))
                    retry_indices.append(i)

                if not retry_prompts:
                    break

                # ========== 🔥【新增打印 3】查看回退重写的情况 ==========
                tee_print(f"\n   🔄 [Retry Round {_retry_idx + 1}] 裁判打回了 {len(retry_prompts)} 条，正在重写...")
                for idx_in_cand, prompt in zip(retry_indices, retry_prompts):
                    # 获取之前的裁判意见
                    last_attempt = candidates[idx_in_cand]["attempts"][-1]
                    
                    # 🔥 修复点：使用 'or {}' 处理 NoneType
                    judge_res = last_attempt.get("judge") or {}
                    
                    # 优化显示：如果没有裁判记录，说明是 Guard 拦截
                    if not judge_res:
                        feedback = "Rejected by Basic Guard (Length/Format)"
                    else:
                        feedback = judge_res.get("feedback", "No feedback")
                        
                    target_mid = candidates[idx_in_cand]["meta"]["mid"][:8]
                    
                    # 防止 feedback 太短导致切片重复显示
                    fb_display = feedback if len(feedback) < 100 else f"{feedback[:50]}...{feedback[-50:]}"
                    tee_print(f"      -> ❌ ID: {target_mid} | 裁判意见: {fb_display}")
                # ======================================================
                
                retry_outs = call_llm_batch(retry_prompts, cfg)
                # ========== 🔥【新增打印 4】查看学生重写结果 ==========
                tee_print(f"   📨 [Retry Output] 收到 {len(retry_outs)} 条重写稿:")
                for i, raw_out in zip(retry_indices, retry_outs):
                    # 这里要通过 i 反查 ID
                    mid = candidates[i]["meta"]["mid"][:8]
                    clean_preview = raw_out.strip().replace('\n', ' ')
                    tee_print(f"      -> 🆔 {mid} | 长度: {len(raw_out)} | 预览: {clean_preview[:20]}...{clean_preview[-20:]}")
                # ====================================================
                for i, raw_out in zip(retry_indices, retry_outs):
                    out_text = _extract_single_memory(raw_out)
                    candidates[i]["attempts"].append({"out_text": out_text, "judge": None})

                judge_items, judge_map = [], []
                for i in retry_indices:
                    meta = candidates[i]["meta"]
                    out_text = candidates[i]["attempts"][-1]["out_text"]
                    if out_text and _basic_guard(out_text, min_len=min_len, max_len=max_len):
                        judge_items.append({
                            "failed_queries": meta.get("neg_text",""),
                            "old_memory": meta.get("old_content",""),
                            "new_memory": out_text,
                        })
                        judge_map.append(i)

                judge_results = _acceptance_test_batch(cfg, judge_items)
                
                # ========== 🔥【新增打印 5】查看重写后的裁判结果 ==========
                if judge_results:
                    tee_print(f"   ⚖️ [Retry Judgment] 收到 {len(judge_results)} 条重审结果:")

                for idx, res in zip(judge_map, judge_results):
                    # 1. 保存结果（原逻辑）
                    candidates[idx]["attempts"][-1]["judge"] = res
                    
                    # 2. 打印日志（新增逻辑）
                    mid = candidates[idx]["meta"]["mid"][:8]
                    verdict = res.get("verdict", "UNKNOWN")
                    feedback = res.get("feedback", "No feedback")
                    
                    # 图标和截断
                    icon = "✅" if verdict == "PASS" else "❌"
                    fb_prev = feedback if len(feedback) < 50 else f"{feedback[:50]}...{feedback[-50:]}"
                    
                    tee_print(f"      -> {icon} ID: {mid} | 结果: {verdict} | 意见: {fb_prev}")
                # ========================================================

            # Commit accepted changes + log
            with open(log_file_path, "a", encoding="utf-8") as log_f:
                for cand in candidates:
                    meta = cand["meta"]
                    info = meta.get("log", {}) or {}
                    target_mid = meta.get("mid")
                    task_type = meta.get("type")

                    # Choose last PASS attempt if needed
                    chosen = None
                    if _need_accept(meta):
                        for att in reversed(cand["attempts"]):
                            txt = (att.get("out_text") or "").strip()
                            j = att.get("judge") or {}
                            if txt and _basic_guard(txt, min_len=min_len, max_len=max_len) and j.get("verdict") == "PASS":
                                chosen = att
                                break
                    if chosen is None:
                        chosen = cand["attempts"][-1]

                    chosen_text = (chosen.get("out_text") or "").strip()
                    chosen_judge = chosen.get("judge") or {}

                    accepted = False
                    if chosen_text and _basic_guard(chosen_text, min_len=min_len, max_len=max_len):
                        if _need_accept(meta):
                            accepted = (chosen_judge.get("verdict") == "PASS")
                        else:
                            accepted = True

                    # # Logging
                    # log_f.write("\n" + "=" * 40 + "\n")
                    # log_f.write(f"🆔 Memory ID: {info.get('mid', target_mid)} | Type: {task_type}\n")
                    # if info.get("expert_prompt"):
                    #     log_f.write(f"--- 🧠 Expert Prompt ---\n{info.get('expert_prompt','')}\n\n")
                    # if info.get("expert_output"):
                    #     log_f.write(f"--- 🗣️ Expert Output ---\n{info.get('expert_output','')}\n\n")
                    # if info.get("action") or info.get("gradient"):
                    #     log_f.write(f"--- 📦 Parsed Action ---\nPrimitive: {info.get('action','')}\nGradient: {info.get('gradient','')}\n\n")
                    # if info.get("student_prompt"):
                    #     log_f.write(f"--- 📝 Student Prompt ---\n{info.get('student_prompt','')}\n\n")
                    # for k, att in enumerate(cand["attempts"]):
                    #     log_f.write(f"--- ✨ Attempt {k} Output ---\n{(att.get('out_text') or '')}\n\n")
                    #     j = att.get("judge")
                    #     if j:
                    #         log_f.write(f"--- ✅ Acceptance (Attempt {k}) ---\nVerdict: {j.get('verdict')}\nFeedback: {j.get('feedback')}\n\n")
                    # log_f.write(f"--- 🧾 Final Decision ---\nAccepted: {accepted}\n")
                    # log_f.write("=" * 40 + "\n")
                    # log_f.flush()

                    if not accepted:
                        continue

                    if task_type in ["refine", "replace"]:
                        if target_mid in memories:
                            memories[target_mid]["contents"] = chosen_text
                            memories[target_mid]["cluster_id"] = -1
                            memories[target_mid]["opt_type"] = meta.get("opt_type", "textgrad")
                            optimized_ids.add(target_mid)

                    elif task_type == "create":
                        print(f"  ✨ [EXPAND] 正在分裂产生新记忆 ID: {target_mid[:8]}...")
                        memories[target_mid] = {
                            "id": target_mid,
                            "contents": chosen_text,
                            "cluster_id": -1,
                            "opt_type": meta.get("opt_type", "textgrad_expand"),
                            "parent_id": meta.get("parent_mid"),
                        }
                        memory_stats[target_mid] = {
                            "alpha": 1.0,
                            "beta": 1.0,
                            "neg_queries": [],
                            "pos_queries": [],
                        }
                        optimized_ids.add(target_mid)
    return optimized_ids