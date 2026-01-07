import os
import re
import uuid
from typing import Set, List, Dict, Optional,Tuple
from omegaconf import DictConfig

# Tooling
from tools.optimize.callllm import call_llm_batch
from tools.optimize.callexpert import call_expert_batch
from utils.memorywrap import parse_memory

_MEMORY_START = r"\\memory{"

def _find_memory_spans(raw_output: str) -> List[Tuple[int, int, str]]:
    """Return list of (start_idx, end_idx_exclusive, inner_text) for each \\memory{...} block.
    Uses brace-depth counting so it won't break on nested braces (e.g., LaTeX \\frac{1}{2}).
    """
    if not raw_output:
        return []
    spans: List[Tuple[int, int, str]] = []
    i = 0
    n = len(raw_output)
    while i < n:
        j = raw_output.find(_MEMORY_START, i)
        if j < 0:
            break
        k = j + len(_MEMORY_START)
        depth = 1
        inner_chars: List[str] = []
        prev = ""
        while k < n and depth > 0:
            ch = raw_output[k]
            # Treat escaped braces (\\{, \\}) as literals w.r.t. depth
            if ch == "{" and prev != "\\":
                depth += 1
            elif ch == "}" and prev != "\\":
                depth -= 1
                if depth == 0:
                    k += 1  # include closing brace
                    break
            if depth > 0:
                inner_chars.append(ch)
            prev = ch
            k += 1
        inner = "".join(inner_chars).strip()
        spans.append((j, k, inner))
        i = max(k, j + 1)
    return spans

def _extract_memory_blocks(raw_output: str) -> List[str]:
    """Extract one or more memory blocks from raw model output.

    Returns a list of *contents* (without the wrapper). Falls back to parse_memory
    if no blocks are found.
    """
    spans = _find_memory_spans(raw_output or "")
    blocks = [s[2].strip() for s in spans if s[2] and s[2].strip()]
    if blocks:
        return blocks
    # Fallback: legacy parser (usually returns a single block)
    single = (parse_memory(raw_output) or "").strip()
    return [single] if single else []

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
# Acceptance test (for high-score evolve candidates)
# ------------------------------------------------------------------------------

_EVOLVE_ACCEPT_PROMPT = r'''
You are a strict evaluator for adding NEW memories to a RAG memory store.

[Failed Queries] (optional; may be empty)
{failed_queries}

[Parent Memory]
{old_memory}

[Candidate New Memory]
{new_memory}

[Task]
Decide whether the Candidate New Memory is (1) relevant, (2) atomic, and (3) likely to improve answers for the Failed Queries without introducing hallucinations or redundant fluff.

[Output Format - STRICT]
Verdict: PASS|FAIL
Feedback: <If FAIL, 1-2 short sentences. If PASS, write "OK".>
'''

_EVOLVE_VERDICT_RE = re.compile(r"Verdict:\s*(PASS|FAIL)", re.IGNORECASE)
_EVOLVE_FEEDBACK_RE = re.compile(r"Feedback:\s*(.*)", re.IGNORECASE | re.DOTALL)

def _evolve_parse_acceptance(output: str):
    if not output:
        return {"verdict": "FAIL", "feedback": "No judge output."}
    m = _EVOLVE_VERDICT_RE.search(output)
    verdict = (m.group(1).upper() if m else "FAIL")
    m2 = _EVOLVE_FEEDBACK_RE.search(output)
    feedback = (m2.group(1).strip() if m2 else "").strip()
    if not feedback:
        feedback = "OK" if verdict == "PASS" else "Missing feedback."
    return {"verdict": verdict, "feedback": feedback}

def _evolve_acceptance_batch(cfg, items):
    prompts = []
    for it in items:
        prompts.append(_EVOLVE_ACCEPT_PROMPT.format(
            failed_queries=(it.get("failed_queries","") or "").strip(),
            old_memory=(it.get("old_memory","") or "").strip(),
            new_memory=(it.get("new_memory","") or "").strip(),
        ))
    if not prompts:
        return []
    outs = call_expert_batch(prompts, cfg)
    return [_evolve_parse_acceptance(o) for o in outs]

def _evolve_acceptance_enabled(cfg) -> bool:
    opt = getattr(cfg, "optimizer", None)
    if opt is None:
        return True
    acc = getattr(opt, "acceptance", None)
    if acc is None:
        return bool(getattr(opt, "acceptance_enabled", True))
    return bool(getattr(acc, "enabled", True))

# ------------------------------------------------------------------------------
# Rollback / retry logic (shared for SUPPLEMENT and SPLIT)
# ------------------------------------------------------------------------------

def _get_max_retries(cfg) -> int:
    try:
        return int(getattr(cfg.parameters, "max_retries", 2) or 2)
    except Exception:
        return int(getattr(cfg.parameters, "max_retries", 2) or 2)

def _judge_one_candidate(cfg, *, failed_queries: str, old_memory: str, new_memory: str) -> Optional[str]:
    """Return None if PASS; else return feedback string."""
    if not _evolve_acceptance_enabled(cfg):
        return None
    if not (failed_queries or "").strip():
        return None
    res = _evolve_acceptance_batch(cfg, [{
        "failed_queries": failed_queries,
        "old_memory": old_memory,
        "new_memory": new_memory,
    }])[0]
    if (res.get("verdict") or "").upper() == "PASS":
        return None
    return res.get("feedback", "Acceptance FAIL.")

def _pick_one_valid(cfg, *, blocks: List[str], failed_queries: str, old_memory: str) -> Tuple[Optional[str], str]:
    """Pick the first candidate that passes guard+acceptance.

    Returns (picked_or_none, failure_reason_for_retry).
    """
    if not blocks:
        return None, "No \\memory{...} block parsed."
    min_l = int(getattr(cfg.optimizer, "min_memory_len", 20) or 20)
    max_l = int(getattr(cfg.optimizer, "max_memory_len", 2000) or 2000)

    first_failure = "Unknown error"
    for cand in blocks[:1]:  # IMPORTANT: always ONE memory (SUPPLEMENT & SPLIT are 1-to-1)
        if not _basic_guard(cand, min_len=min_l, max_len=max_l):
            first_failure = "Rejected by basic guard (length or banned words)."
            continue
        fb = _judge_one_candidate(cfg, failed_queries=failed_queries, old_memory=old_memory, new_memory=cand)
        if fb is None:
            return cand, "OK"
        first_failure = fb
    return None, first_failure

def _rewrite_with_feedback(cfg, *, base_prompt: str, prev_output: str, feedback: str) -> str:
    retry_prompt = (
        base_prompt
        + "\n\n[Previous Attempt]\n"
        + (prev_output or "")[:800]
        + "\n\n[Judge Feedback]\n"
        + (feedback or "")
        + "\n\nRewrite following the feedback. Output ONLY ONE \\memory{...} block."
    )
    return call_llm_batch([retry_prompt], cfg)[0]

# ------------------------------------------------------------------------------
# Main high-score evolution
# ------------------------------------------------------------------------------

def evolve_high_score_opt(cfg: DictConfig, memories: Dict, memory_stats: Dict, high_ids: List[str]) -> Set[str]:
    """High-score memory evolution (SUPPLEMENT / SPLIT) with robust parsing & rollback.
    Semantics: both SUPPLEMENT and SPLIT add exactly ONE new memory; champion memory is never modified.
    """
    print("\n========== 高分记忆进化阶段 (Ace Evolution) ==========")

    # --- 1) Log path ---
    log_file_path = cfg.paths.get("highfreq_textgrad_log", "textgrad_debug_log.txt")
    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    except Exception:
        pass
    print(f"📝 进化日志将追加至: {log_file_path}")

    def tee_print(msg):
        """同时打印到终端并追加写入日志文件"""
        # print(msg) # 打印到屏幕
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n") # 写入文件
        except Exception:
            pass

    # --- 2) Target selection: only champions with failed queries ---
    target_ids: List[str] = []
    for mid in list(high_ids):
        if mid not in memories:
            continue
        stats = memory_stats.get(mid, {}) or {}
        neg_queries = stats.get("neg_queries", []) or []
        if len(neg_queries) > 0:
            target_ids.append(mid)

    # Optional debug limit
    debug_limit = int(getattr(cfg.optimizer, "debug_high_score_limit", 0) or 0)
    if debug_limit > 0:
        target_ids = target_ids[:debug_limit]

    print(f"💎 待进化的王牌记忆数量: {len(target_ids)}")
    if target_ids:
        print(f"   🆔 ID 列表: {target_ids}")
    if not target_ids:
        return set()

    batch_size = int(cfg.optimizer.llm_batch_size)
    new_created_ids_total: Set[str] = set()

    for i in range(0, len(target_ids), batch_size):
        chunk_ids = target_ids[i : i + batch_size]
        print(f" 🧠 [Expert-Batch] 正在处理第 {i} - {i+len(chunk_ids)} 条高分记忆...")

        expert_prompts: List[str] = []
        chunk_metadata: List[dict] = []

        for mid in chunk_ids:
            rec = memories[mid]
            base_text = rec.get("contents", "")
            stats = memory_stats.get(mid, {}) or {}
            neg_queries = stats.get("neg_queries", []) or []

            top_k_neg = int(getattr(cfg.optimizer, "high_grad_topk_neg", 5) or 5)
            neg_text = "\n".join([f"- {q}" for q in neg_queries[:top_k_neg]])

            try:
                prompt = cfg.optimizer.prompts.high_grad_expert.format(content=base_text, neg_queries=neg_text)
            except Exception as e:
                print(f"❌ Prompt 格式化失败 (MID: {mid}): {e}")
                continue

            expert_prompts.append(prompt)
            chunk_metadata.append({
                "mid": mid,
                "base_text": base_text,
                "neg_text": neg_text,
                "expert_prompt_content": prompt,
            })

        if not expert_prompts:
            continue

        expert_outputs = call_expert_batch(expert_prompts, cfg)

        student_prompts: List[str] = []
        student_tasks: List[dict] = []

        for meta, expert_resp in zip(chunk_metadata, expert_outputs):
            mid = meta["mid"]

            log_info = {
                "mid": mid,
                "type": "evolve_high_score",
                "expert_prompt": meta.get("expert_prompt_content", ""),
                "expert_output": expert_resp,
                "action": "UNKNOWN",
                "gradient": "N/A",
                "split_num": 1,  # kept for backward compatibility in logs
                "student_prompt": "N/A",
            }

            if not expert_resp:
                tee_print(f"❌ [Error] MID: {mid} - Expert output is empty.")
                continue

            action_match = re.search(r"\\box\{(IGNORE|SUPPLEMENT|SPLIT)\}", expert_resp)
            action = action_match.group(1).strip() if action_match else "IGNORE"

            gradient_match = re.search(r"\\gradient\{(.*?)\}", expert_resp, re.DOTALL)
            advice = gradient_match.group(1).strip() if gradient_match else "No specific advice provided."
            gradient = advice
            # ================= 打印代码 =================
            tee_print(f"\n[Expert Logic] MID: {mid}")
            tee_print(f"   >>> 🛠️ 原语 (Action): {action}")
            tee_print(f"   >>> 🧠 梯度 (Gradient): {gradient[:20]}...{gradient[-20:]}") # 只打印40字
            # ====================================================

            # IMPORTANT: SPLIT is forced to ONE new memory (no 1-to-many)
            split_num = 1

            log_info["action"] = action
            log_info["gradient"] = advice
            log_info["split_num"] = split_num

            if action == "IGNORE":
                with open(log_file_path, "a", encoding="utf-8") as log_f:
                    mid_display = str(mid)[:8]
                    # 截断 Advice/Gradient 显示
                    grad = str(advice)
                    grad_prev = f"{grad[:20]}...{grad[-20:]}" if len(grad) > 40 else grad
                    
                    log_lines = [
                        f"🆔 [{mid_display}] | EVOLVE | 🚫 IGNORED",
                        f"   Strategy: High-Score-Evolve",
                        f"   Action  : IGNORE",
                        f"   Reason  : {grad_prev}",  # 这里的 advice 通常包含为什么忽略的原因
                        "-" * 60 + "\n"
                    ]
                    log_f.write("\n".join(log_lines))
                    log_f.flush()
                # ==============================================================
                continue

            if action == "SUPPLEMENT":
                tpl = cfg.optimizer.prompts.appgrad_high_supplement
                s_prompt = tpl.format(original_content=meta["base_text"], advice=advice)
                log_info["student_prompt"] = s_prompt
                student_prompts.append(s_prompt)
                student_tasks.append({
                    "parent_mid": mid,
                    "action": "SUPPLEMENT",
                    "log": log_info,
                    "old_content": meta.get("base_text",""),
                    "neg_text": meta.get("neg_text",""),
                })

            elif action == "SPLIT":
                tpl = cfg.optimizer.prompts.appgrad_high_split
                # allow templates with/without {num}
                try:
                    s_prompt = tpl.format(neg_text=meta["neg_text"], advice=advice, num=1)
                except Exception:
                    s_prompt = tpl.format(neg_text=meta["neg_text"], advice=advice)
                log_info["student_prompt"] = s_prompt
                student_prompts.append(s_prompt)
                student_tasks.append({
                    "parent_mid": mid,
                    "action": "SPLIT",
                    "log": log_info,
                    "old_content": meta.get("base_text",""),
                    "neg_text": meta.get("neg_text",""),
                })

        if not student_prompts:
            continue

        # ========== 🔥【新增打印 1】查看学生领到的任务单 ==========
        # 适配说明：原变量 student_metadata 在这里对应 student_tasks
        tee_print(f"   📋 学生任务详情:")
        for task in student_tasks:
            # 获取动作类型 (SUPPLEMENT/SPLIT)
            action_type = task.get("action", "UNKNOWN").upper()
            # 获取 ID
            mid_display = task['parent_mid'][:8] 
            
            tee_print(f"   -> 🆔 [{mid_display}] | 动作: {action_type} | 策略: High-Score-Evolve")
        tee_print("   ------------------------------------------------")
        # =======================================================

        student_outputs = call_llm_batch(student_prompts, cfg)

        # ========== 🔥【新增打印 2】查看学生初稿 ==========
        # 适配说明：outputs 对应 student_outputs
        tee_print(f"\n   📨 [Student Output] 收到 {len(student_outputs)} 条初稿:")
        for task, raw_txt in zip(student_tasks, student_outputs):
            clean_preview = raw_txt.strip().replace('\n', ' ')
            mid = task['parent_mid'][:8]
            # 防止字符串切片报错，做个简单长度判断
            preview_str = clean_preview if len(clean_preview) < 40 else f"{clean_preview[:20]}...{clean_preview[-20:]}"
            tee_print(f"      -> 🆔 {mid} | 长度: {len(raw_txt)} | 预览: {preview_str}")
        # =================================================

        max_retries = _get_max_retries(cfg)

        # ==============================================================================
        # 🔥 核心优化：并行化验证与重写流程 (Parallel Validation & Rewrite)
        # ==============================================================================
        
        # 1. 初始化候选状态列表
        candidates = []
        for task, raw_out in zip(student_tasks, student_outputs):
            candidates.append({
                "task": task,
                "history": [{"out": raw_out, "judge": None}],
                "status": "PENDING", # PENDING, PASS, FAIL
                "final_output": None,
                "fail_reason": ""
            })

        max_retries = _get_max_retries(cfg)
        min_len = int(getattr(cfg.optimizer, "min_memory_len", 20) or 20)
        max_len = int(getattr(cfg.optimizer, "max_memory_len", 2000) or 2000)

        # 2. 批处理循环 (Batch Loop): 验证 -> 筛选失败者 -> 批量重写 -> 再次验证
        # 循环次数 = 初始验证(Round 0) + 重试次数(max_retries)
        
        for round_idx in range(max_retries + 1):
            
            # --- Step A: 批量验证 (Batch Judge) ---
            to_judge_indices = []
            judge_payloads = []
            
            for i, cand in enumerate(candidates):
                # 只处理状态为 PENDING 的任务 (即尚未通过且有新输出的任务)
                if cand["status"] != "PENDING":
                    continue
                    
                last_attempt = cand["history"][-1]
                raw_txt = last_attempt["out"]
                
                # 1. 提取 Memory Block
                blocks = _extract_memory_blocks(raw_txt)
                
                # 2. 基础 Guard 检查 (长度/违禁词)
                valid_block = None
                for b in blocks:
                    if _basic_guard(b, min_len=min_len, max_len=max_len):
                        valid_block = b
                        break
                
                if not valid_block:
                    cand["fail_reason"] = "Rejected by basic guard (length/format/banned)."
                    last_attempt["judge"] = {"verdict": "FAIL", "feedback": cand["fail_reason"]}
                    # 状态保持 PENDING，留给重写阶段处理
                else:
                    last_attempt["parsed_block"] = valid_block # 暂存合法的块
                    
                    # 3. 决定是否需要裁判 (有错题才需要 Expert Judge)
                    meta = cand["task"]
                    neg_text = meta.get("neg_text", "")
                    
                    if _evolve_acceptance_enabled(cfg) and neg_text:
                        to_judge_indices.append(i)
                        judge_payloads.append({
                            "failed_queries": neg_text,
                            "old_memory": meta.get("old_content", ""),
                            "new_memory": valid_block
                        })
                    else:
                        # 没开裁判或没有错题 -> 直接 PASS
                        cand["status"] = "PASS"
                        cand["final_output"] = valid_block
                        last_attempt["judge"] = {"verdict": "PASS", "feedback": "OK (Skipped)"}

            # --- 发送批量裁判请求 (真正的并行验证) ---
            if judge_payloads:
                tee_print(f"   ⚖️ [Batch Judge] Round {round_idx}: 正在评审 {len(judge_payloads)} 条候选项...")
                # 这里的 _evolve_acceptance_batch 内部会调用 call_expert_batch，享受你刚才改的 16 并发
                judge_results = _evolve_acceptance_batch(cfg, judge_payloads)
                
                for idx, res in zip(to_judge_indices, judge_results):
                    cand = candidates[idx]
                    last_attempt = cand["history"][-1]
                    last_attempt["judge"] = res
                    
                    if res["verdict"] == "PASS":
                        cand["status"] = "PASS"
                        cand["final_output"] = last_attempt["parsed_block"]
                    else:
                        cand["fail_reason"] = res["feedback"]
                        # 状态保持 PENDING

            # --- Step B: 准备批量重写 (Batch Rewrite) ---
            # 如果是最后一次循环，就不重写了，直接结束
            if round_idx == max_retries:
                break

            to_rewrite_indices = []
            retry_prompts = []

            for i, cand in enumerate(candidates):
                if cand["status"] == "PENDING":
                    to_rewrite_indices.append(i)
                    
                    # 构造重写 Prompt
                    last_attempt = cand["history"][-1]
                    prev_out = last_attempt.get("out", "")
                    fb = cand.get("fail_reason", "Improve logic.")
                    base_prompt = cand["task"]["log"].get("student_prompt", "")
                    
                    # 构造重写提示词
                    retry_prompt = (
                        base_prompt
                        + "\n\n[Previous Attempt]\n"
                        + (prev_out or "")[:800]
                        + "\n\n[Judge Feedback]\n"
                        + (fb or "")
                        + "\n\nRewrite following the feedback. Output ONLY ONE \\memory{...} block."
                    )
                    retry_prompts.append(retry_prompt)

            if not retry_prompts:
                # 所有任务都处理完了 (全PASS或没得救了)
                break
            
            # --- 发送批量重写请求 (真正的并行重写) ---
            tee_print(f"\n   🔄 [Batch Retry] Round {round_idx+1}: {len(retry_prompts)} 条任务被打回，正在批量重写...")
            
            # 简单打印前几个失败原因供调试
            for idx in to_rewrite_indices[:2]:
                    mid = candidates[idx]["task"]["parent_mid"][:8]
                    fb = candidates[idx].get("fail_reason", "")
                    tee_print(f"      -> ❌ ID: {mid} | 原因: {fb[:50]}...")

            # 这里的 call_llm_batch 内部是你刚才改的 SGLang 并发版，享受 32 并发
            retry_outputs = call_llm_batch(retry_prompts, cfg)
            
            # 填入新结果，等待下一轮验证
            for idx, new_out in zip(to_rewrite_indices, retry_outputs):
                # 打印一下重写结果预览
                mid = candidates[idx]["task"]["parent_mid"][:8]
                clean = new_out.strip().replace('\n', ' ')
                prev = clean if len(clean) < 40 else f"{clean[:20]}...{clean[-20:]}"
                tee_print(f"      -> 📨 [Retry] ID: {mid} | 预览: {prev}")
                
                candidates[idx]["history"].append({"out": new_out, "judge": None})

        # ==============================================================================
        # 3. 最终结算与日志保存 (Finalize)
        # ==============================================================================
        for cand in candidates:
            task = cand["task"]
            log_info = task["log"]
            parent_mid = task["parent_mid"]
            action_type = task["action"]
            
            # 判定最终状态
            accepted = (cand["status"] == "PASS" and cand["final_output"] is not None)
            
            # 准备日志内容
            final_txt = cand["final_output"] if accepted else cand["history"][-1]["out"]
            final_txt = (final_txt or "").strip().replace('\n', ' ')
            content_prev = f"{final_txt[:30]}...{final_txt[-30:]}" if len(final_txt) > 60 else final_txt
            
            last_judge = cand["history"][-1].get("judge") or {}
            judge_verdict = last_judge.get("verdict", "FAIL")
            judge_fb = last_judge.get("feedback", cand.get("fail_reason", "Unknown"))
            judge_prev = f"{judge_fb[:50]}..." if len(judge_fb) > 50 else judge_fb
            
            # 写入简洁日志
            with open(log_file_path, "a", encoding="utf-8") as log_f:
                status_str = "✅ ACCEPTED" if accepted else "❌ REJECTED"
                grad = str(log_info.get("gradient", ""))
                grad_prev = f"{grad[:20]}...{grad[-20:]}" if len(grad) > 40 else grad
                
                log_lines = [
                    f"🆔 [{parent_mid[:8]}] | {action_type} | {status_str}",
                    f"   Strategy: High-Score-Evolve (Batched)",
                    f"   Action  : {action_type}",
                    f"   Gradient: {grad_prev}",
                    f"   Result  : {content_prev}",
                    f"   Judge   : {judge_verdict} ({judge_prev})",
                    "-" * 60 + "\n"
                ]
                log_f.write("\n".join(log_lines))
                log_f.flush()

            # 如果成功，保存到内存库
            if accepted:
                new_id = str(uuid.uuid4())
                suffix = "supplement" if action_type == "SUPPLEMENT" else "split"
                _save_new_memory(memories, memory_stats, new_id, cand["final_output"], parent_mid, f"high_score_{suffix}")
                new_created_ids_total.add(new_id)
                tee_print(f"  ✨ [NEW] {parent_mid[:8]} -> {new_id[:8]} ({action_type})")

    print(f"✅ [Evolve] 进化完成，共新增 {len(new_created_ids_total)} 条高阶记忆")
    return new_created_ids_total

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _save_new_memory(memories, memory_stats, new_id, content, parent_id, opt_type):
    memories[new_id] = {
        "id": new_id,
        "contents": content,
        "cluster_id": -1,
        "opt_type": opt_type,
        "parent_id": parent_id,
    }
    memory_stats[new_id] = {
        "alpha": 1.0,
        "beta": 1.0,
        "neg_queries": [],
        "pos_queries": [],
    }

def _write_log(log_file_path: str, info: dict, result_content: str):
    try:
        with open(log_file_path, "a", encoding="utf-8") as f:
            log_entry = (
                f"\n{'='*60}\n"
                f"🆔 Parent Memory ID: {info.get('mid','')}\n"
                f"--- 🧠 Expert Prompt (Input) ---\n{info.get('expert_prompt','')}\n\n"
                f"--- 🗣️ Expert Output (Raw) ---\n{info.get('expert_output','')}\n\n"
                f"--- 📦 Parsed Decision ---\n"
                f"   Action   : {info.get('action','')}\n"
                f"   Advice   : {info.get('gradient','')}\n"
                f"   Split Num: {info.get('split_num',1)}\n\n"
                f"--- 📝 Student Prompt ---\n{info.get('student_prompt','')}\n\n"
                f"--- ✨ Final Result (New Memories) ---\n{result_content}\n"
                f"{'='*60}\n"
            )
            f.write(log_entry)
            f.flush()
    except Exception as e:
        print(f"⚠️ 日志写入异常: {e}")