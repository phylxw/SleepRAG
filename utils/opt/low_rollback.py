import re
# 假设这是你的工具库路径
from tools.optimize.callexpert import call_expert_batch


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
    prev = (prev_memory or "").strip()
    fb = (judge_feedback or "").strip()
    return (
        original_student_prompt
        + "\n\n[Previous Attempt]\n"
        + "\\memory$\n"
        + prev
        + "\n\\endmemory$\n\n"
        + "[Judge Feedback]\n"
        + fb
        + "\n\nRewrite again. Output ONLY:\n\\memory$\n<content>\n\\endmemory$"
    )
