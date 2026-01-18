import os
import re
import ast
import json
import builtins
import keyword
import textwrap
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional


class CodeEvaluator:
    """
    Code Evaluator v3.2
    Key fixes:
      - MBPP: NO exec() for tests (sandbox-friendly)
      - Always log full server JSON on failure (schema-agnostic)
      - Print traceback to stdout with markers (even if stderr not returned)
      - Safer code extraction: dedent + rstrip
      - Safer MBPP function-name alignment: AST-based + forbid overriding builtins/keywords
      - Optional: wrap "function-body-only" outputs into a def shell for MBPP
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        max_workers: int = 8,
        timeout: Tuple[float, float] = (3.0, 30.0),  # (connect, read)
        inject_common_imports: bool = True,
        verbose_fail_json_chars: int = 2000,
    ):
        self.server_url = server_url.rstrip("/")
        self.max_workers = max_workers
        self.timeout = timeout
        self.inject_common_imports = inject_common_imports
        self.verbose_fail_json_chars = verbose_fail_json_chars

        self._builtin_names = set(dir(builtins))
        self._keyword_names = set(keyword.kwlist)

        # Denylist: never treat these as "target function name"
        self._deny_names = set([
            # very common builtins
            "len", "range", "print", "map", "filter", "zip", "sum", "max", "min",
            "sorted", "set", "list", "tuple", "dict", "int", "float", "str", "bool",
            "abs", "all", "any", "enumerate", "reversed",
            # testing
            "assert",
        ]) | self._builtin_names | self._keyword_names

    # -------------------------
    # 1) Code extraction
    # -------------------------
    def extract_python_code(self, text: str) -> str:
        """Extract python code from markdown. Keep relative indentation."""
        if text is None:
            return ""

        # Prefer ```python ... ```
        m = re.search(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            return textwrap.dedent(m.group(1)).rstrip()

        # Fallback ``` ... ```
        m = re.search(r"```(.*?)```", text, re.DOTALL)
        if m:
            return textwrap.dedent(m.group(1)).rstrip()

        return text.strip()

    # -------------------------
    # 2) Imports header
    # -------------------------
    def _build_header_imports(self, code_body: str) -> str:
        if not self.inject_common_imports:
            return ""

        common_imports = [
            "import math",
            "import re",
            "import sys",
            "import collections",
            "import heapq",
            "import itertools",
            "from typing import *",
        ]

        header = []
        lower = code_body.lower()
        for imp in common_imports:
            if imp.startswith("import "):
                token = imp.split()[1]
                if f"import {token}" in lower:
                    continue
            elif imp.startswith("from "):
                token = imp.split()[1]
                if f"from {token} import" in lower:
                    continue
            header.append(imp)

        return ("\n".join(header) + "\n") if header else ""

    # -------------------------
    # 3) AST helpers
    # -------------------------
    def _defined_functions(self, code_body: str) -> List[str]:
        names: List[str] = []
        try:
            tree = ast.parse(code_body)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    names.append(node.name)
        except Exception:
            names = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code_body)
        return names

    def _infer_mbpp_target_name_from_tests(self, test_list: List[str]) -> Optional[str]:
        """
        Infer "target function name" from MBPP tests:
          - Parse each test string with AST if possible
          - Count ast.Call where func is ast.Name (foo(...))
          - Skip builtins/keywords/denylist
        """
        counts: Dict[str, int] = {}
        for t in test_list:
            if not isinstance(t, str) or not t.strip():
                continue
            try:
                tree = ast.parse(t)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        name = node.func.id
                        if name in self._deny_names:
                            continue
                        counts[name] = counts.get(name, 0) + 1
            except Exception:
                # fallback regex
                for name in re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", t):
                    if name in self._deny_names:
                        continue
                    counts[name] = counts.get(name, 0) + 1

        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _maybe_make_alignment_code(
        self,
        defined_funcs: List[str],
        target_name: Optional[str],
        code_body: str,
    ) -> str:
        """
        Safe alignment:
          - Only if target_name exists AND target_name not defined
          - If only 1 function defined => alias target = that function
          - If multiple => alias to last defined (common CoT), still conservative
        """
        if not target_name:
            return ""
        if target_name in defined_funcs:
            return ""
        if not defined_funcs:
            return ""

        # If code already assigns target_name, skip
        if re.search(rf"\b{re.escape(target_name)}\b\s*=", code_body):
            return ""

        if len(defined_funcs) == 1:
            src = defined_funcs[0]
            if src != target_name:
                return f"\n{target_name} = {src}\n"
            return ""

        src = defined_funcs[-1]
        if src != target_name:
            return f"\n{target_name} = {src}\n"
        return ""

    def _maybe_wrap_function_body(self, code_body: str, target_name: Optional[str]) -> str:
        """
        If model outputs only function body (no def), MBPP will SyntaxError.
        Conservative wrap:
          def target(*args, **kwargs):
              <body>
        """
        if "def " in code_body:
            return code_body
        if not target_name:
            return code_body

        body = textwrap.dedent(code_body).strip("\n")
        if not body:
            return code_body
        body_ind = textwrap.indent(body, "    ")
        return f"def {target_name}(*args, **kwargs):\n{body_ind}\n"

    # -------------------------
    # 4) Build runnable code
    # -------------------------
    def _wrap_with_traceback_stdout(self, inner_code: str) -> str:
        """
        Wrap execution so that any exception prints traceback into stdout with markers,
        then re-raises to make runner mark it as failure.
        """
        return f"""
try:
{textwrap.indent(inner_code, "    ")}
except Exception:
    import traceback as __tb
    print("EVAL_ERROR_TRACEBACK_BEGIN")
    print(__tb.format_exc())
    print("EVAL_ERROR_TRACEBACK_END")
    raise
"""

    def _build_humaneval_code(self, code_body: str, task_data: Dict[str, Any]) -> Tuple[str, str]:
        entry_point = task_data.get("entry_point", "candidate")
        test_code = task_data.get("test", "")
        prompt = task_data.get("prompt", "")

        if not isinstance(test_code, str) or not test_code.strip():
            return "", "Missing HumanEval test code"

        header = self._build_header_imports(code_body)

        inner = (
            f"{header}\n"
            f"{prompt}\n"
            f"{code_body}\n\n"
            f"candidate = {entry_point}\n"
            f"{test_code}\n\n"
            f"check(candidate)\n"
        )
        full_code = self._wrap_with_traceback_stdout(inner)
        return full_code, ""

    def _build_mbpp_code(self, code_body: str, task_data: Dict[str, Any]) -> Tuple[str, str]:
        test_list = task_data.get("test_list", [])
        setup_code = task_data.get("test_setup_code", "")

        if not isinstance(test_list, list) or len(test_list) == 0:
            return "", "Missing MBPP test_list"

        target_name = self._infer_mbpp_target_name_from_tests(test_list)

        # Wrap if needed (no def)
        code_body2 = self._maybe_wrap_function_body(code_body, target_name)

        defined_funcs = self._defined_functions(code_body2)
        alignment_code = self._maybe_make_alignment_code(defined_funcs, target_name, code_body2)

        header = self._build_header_imports(code_body2)

        # Build sandbox-friendly test runner:
        # NO exec(string). We insert test statements directly into try blocks.
        runner_lines = []
        runner_lines.append("def __run_mbpp_tests():")
        for i, tc in enumerate(test_list):
            if not isinstance(tc, str) or not tc.strip():
                continue
            tc_clean = tc.strip("\n")
            tc_block = "\n".join(("        " + line) for line in tc_clean.splitlines())
            runner_lines.append("    try:")
            runner_lines.append(tc_block)
            runner_lines.append("    except Exception as __e:")
            safe_tc = tc_clean.replace('"', '\\"')
            # 2. 然后再拼接
            runner_lines.append(
                "        raise RuntimeError("
                f"\"MBPP Test Failed\\nTest ID: {i}\\nTest: {safe_tc}\\n\""
                " + f\"Error: {type(__e).__name__}: {__e}\""
                "        ) from __e"
            )
        runner_lines.append("__run_mbpp_tests()")
        runner_code = "\n".join(runner_lines) + "\n"

        inner = (
            f"{header}\n"
            f"{setup_code}\n"
            f"{code_body2}\n"
            f"{alignment_code}\n"
            f"{runner_code}\n"
        )
        full_code = self._wrap_with_traceback_stdout(inner)
        return full_code, ""

    # -------------------------
    # 5) Run via server (schema-agnostic)
    # -------------------------
    def _summarize_fail_json(self, res_json: Dict[str, Any]) -> str:
        try:
            s = json.dumps(res_json, ensure_ascii=False)
        except Exception:
            s = str(res_json)
        if len(s) > self.verbose_fail_json_chars:
            s = s[: self.verbose_fail_json_chars] + " ...[truncated]..."
        return s

    def _run_remote(self, full_code: str) -> Tuple[bool, str, Dict[str, Any]]:
        try:
            resp = requests.post(
                f"{self.server_url}/run_code",
                json={"code": full_code, "language": "python"},
                timeout=self.timeout,
            )
        except Exception as e:
            return False, f"RequestError: {type(e).__name__}: {e}", {}

        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}: {resp.text[:1000]}", {}

        try:
            res_json = resp.json()
        except Exception:
            return False, f"BadJSONResponse: {resp.text[:2000]}", {}

        status = str(res_json.get("status", ""))

        if status == "Success":
            return True, "Success", res_json

        # Try to find any error-like fields (unknown schema)
        # Common candidates: error/stderr/traceback/message/result/output
        candidates = []
        for k in ["error", "stderr", "traceback", "exception", "message", "result", "output", "details"]:
            v = res_json.get(k, None)
            if v:
                candidates.append(f"{k}={str(v)[:800]}")
        stdout = res_json.get("stdout", "")
        if stdout:
            candidates.append(f"stdout={str(stdout)[:800]}")

        # If still empty, include full json (truncated)
        if not candidates:
            candidates.append("full_json=" + self._summarize_fail_json(res_json))

        reason = f"RunnerFail: status={status}; " + "; ".join(candidates)
        return False, reason, res_json

    # -------------------------
    # 6) Public APIs
    # -------------------------
    def evaluate_one(self, dataset_type: str, pred_str: str, task_data: Dict[str, Any]) -> Tuple[float, str]:
        code_body = self.extract_python_code(pred_str)
        if not code_body.strip():
            return 0.0, "Empty code"

        if dataset_type == "humaneval":
            full_code, build_err = self._build_humaneval_code(code_body, task_data)
        elif dataset_type == "mbpp":
            full_code, build_err = self._build_mbpp_code(code_body, task_data)
        else:
            return 0.0, f"Unknown dataset_type={dataset_type}"

        if build_err:
            return 0.0, f"BuildError: {build_err}"
        if not full_code.strip():
            return 0.0, "Empty full_code"

        ok, reason, _payload = self._run_remote(full_code)
        return (1.0, "") if ok else (0.0, reason)

    def evaluate_batch(
        self,
        dataset_type: str,
        pred_list: List[str],
        task_data_list: List[Dict[str, Any]],
    ) -> Tuple[List[float], List[str]]:
        n = len(pred_list)
        scores = [0.0] * n
        reasons = [""] * n

        print(f"⚖️ [CodeEval] Evaluating {n} samples (dataset={dataset_type}, workers={self.max_workers})...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self.evaluate_one, dataset_type, pred, item): i
                       for i, (pred, item) in enumerate(zip(pred_list, task_data_list))}

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    s, r = fut.result()
                except Exception as e:
                    s, r = 0.0, f"InternalError: {type(e).__name__}: {e}"
                scores[i] = s
                reasons[i] = r

        return scores, reasons


def evaluate_code_results(
    results: List[Any],
    experiment_name: str,
    result_log_file: str,
    dataset_type: str = "humaneval",
    server_url: str = "http://localhost:8080",
    max_workers: int = 8,
) -> Tuple[float, List[float]]:
    evaluator = CodeEvaluator(server_url=server_url, max_workers=max_workers)

    task_data_list: List[Dict[str, Any]] = []
    preds_list: List[str] = []

    for item in results:
        data_dict = item.__dict__ if hasattr(item, "__dict__") else item
        if not isinstance(data_dict, dict):
            data_dict = {}
        task_data_list.append(data_dict)

        pred = item.pred if hasattr(item, "pred") else (item.get("pred", "") if isinstance(item, dict) else "")
        preds_list.append(pred)

    print(f"🚀 [Eval] Running code evaluation for {len(results)} samples ({dataset_type})...")
    scores, reasons = evaluator.evaluate_batch(dataset_type, preds_list, task_data_list)

    correct = 0
    total = len(results)

    os.makedirs(os.path.dirname(result_log_file) or ".", exist_ok=True)

    with open(result_log_file, "a", encoding="utf-8") as f:
        header = f"\n{'='*20} {experiment_name} (Code) {'='*20}\n"
        print(header.strip())
        f.write(header)

        for i, (item, score, pred, reason) in enumerate(zip(results, scores, preds_list, reasons)):
            is_right = (score == 1.0)
            if is_right:
                correct += 1

            q_text = ""
            if hasattr(item, "prompt"):
                q_text = item.prompt
            elif hasattr(item, "text"):
                q_text = item.text
            elif isinstance(item, dict):
                q_text = item.get("prompt", item.get("text", ""))

            extracted_code = evaluator.extract_python_code(pred)

            log_entry = (
                f"\n[ID]: {i}\n"
                f"[Question/Prompt]: {str(q_text)[:80]}...{str(q_text)[-80:]}\n"
                f"[Pred Extracted]:\n{extracted_code}\n"
                f"[Result]: {'✅ Correct' if is_right else '❌ Wrong (Pass@1)'}\n"
            )
            if not is_right:
                log_entry += f"[Failure Reason]: {reason}\n"
            log_entry += "-" * 30 + "\n"

            f.write(log_entry)

            if i < 1 or (not is_right and i < 15):
                print(log_entry.strip())

        acc = (correct / total * 100) if total > 0 else 0.0
        summary = (
            f"\n📊 Summary ({experiment_name}):\n"
            f"Dataset: {dataset_type.upper()}\n"
            f"Total: {total}, Correct: {correct}, Accuracy (Pass@1): {acc:.2f}%\n"
            f"{'='*50}\n"
        )
        print(summary)
        f.write(summary)

    return acc, scores


# -------------------------
# Self-test
# -------------------------
if __name__ == "__main__":
    print("🚀 [Self-Test] Start...")

    evaluator = CodeEvaluator(server_url="http://localhost:8080", max_workers=2)

    # MBPP-like test: name mismatch (should be fixed by alignment)
    mbpp_task_data = {
        "test_setup_code": "",
        "test_list": [
            "assert get_sqrt(4) == 2.0",
            "assert get_sqrt(9) == 3.0",
        ],
    }
    mbpp_pred = "def my_sqrt(n):\n    import math\n    return math.sqrt(n)\n"
    s, r = evaluator.evaluate_one("mbpp", mbpp_pred, mbpp_task_data)
    print(f"[MBPP] score={s}, reason={r}")
