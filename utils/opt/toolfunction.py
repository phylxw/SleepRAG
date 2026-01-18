import re
from typing import List, Optional, Dict, Any, Tuple
from utils.opt.memorywrap import parse_memory


# ==============================================================================
# 1. 常量与正则 (统一使用低分优化的定义)
# ==============================================================================

# 统一：支持 \memory / \\memory，$ 可选，允许空白
_MEMORY_BLOCK_RE = re.compile(
    r"(?:\\{1,2})memory\$?\s*(.*?)\s*(?:\\{1,2})endmemory\$?\s*",
    re.DOTALL | re.IGNORECASE
)

# 兜底：无论 marker 出现在正文哪，都剥掉
_MARKER_STRIP_RE = re.compile(
    r"(?:\\{1,2})memory\$?|(?:\\{1,2})endmemory\$?",
    re.IGNORECASE
)

# ==============================================================================
# 2. 低分记忆文本处理与提取工具
# ==============================================================================

def clean_memory_text(s: str) -> str:
    """清洗提取出的记忆文本（去除残留标记、压缩空白）。"""
    if not s:
        return ""
    # 全部空白压缩（含真实换行/tab/formfeed）
    s = re.sub(r"\s+", " ", s)
    # 剥掉残留 marker
    s = _MARKER_STRIP_RE.sub("", s)
    return s.strip()

def extract_memory_blocks(raw_output: str) -> List[str]:
    """
    通用提取函数：返回所有找到的 memory 块列表。
    高分优化中如果涉及 SPLIT 操作可能需要返回多个块，这里做通用支持。
    """
    if not raw_output:
        return []

    # 1. 尝试通过正则提取所有块
    blocks = []
    for m in _MEMORY_BLOCK_RE.finditer(raw_output):
        cleaned = clean_memory_text(m.group(1))
        if cleaned:
            blocks.append(cleaned)
    
    if blocks:
        return blocks

    # 2. 如果正则没匹配到，尝试使用外部 parse_memory (低分逻辑)
    # 注意：parse_memory 通常只返回一个字符串
    external_parse = parse_memory(raw_output)
    if external_parse:
        cleaned = clean_memory_text(external_parse)
        if cleaned:
            return [cleaned]

    # 3. 兜底：如果完全没有格式，尝试暴力清洗全文
    fallback = clean_memory_text(raw_output)
    return [fallback] if fallback else []

def _extract_single_memory(raw_output: str) -> str:
    """
    [低分优化常用] 强制只提取一个记忆块。
    如果 LLM 输出了多个，默认取第一个。
    """
    blocks = extract_memory_blocks(raw_output)
    return blocks[0] if blocks else ""

# ==============================================================================
# 3. 高分记忆文本处理与提取工具
# ==============================================================================

def _clean_block(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)         # 压缩真实空白
    s = _MARKER_STRIP_RE.sub("", s)    # 剥离残留 marker
    return s.strip()

def _find_memory_spans(raw_output: str) -> List[Tuple[int, int, str]]:
    """Return list of (start_idx, end_idx_exclusive, inner_text) for each memory block."""
    if not raw_output:
        return []
    spans: List[Tuple[int, int, str]] = []
    for m in _MEMORY_BLOCK_RE.finditer(raw_output): 
        inner = _clean_block(m.group(1))
        spans.append((m.start(), m.end(), inner))
    return spans

def _extract_memory_blocks(raw_output: str) -> List[str]:
    spans = _find_memory_spans(raw_output or "")
    blocks = [inner for (_, _, inner) in spans if inner]
    if blocks:
        return blocks

    # fallback：如果模型没按 wrapper 输出，最后也清洗一下全文，避免 endmemory 残留
    return [_clean_block(raw_output or "")] if (raw_output or "").strip() else []

# ==============================================================================
# 4. 卫栏与安全检查 (Guardrails)
# ==============================================================================

def _basic_guard(text: str, *, min_len: int = 20, max_len: int = 2000) -> bool:
    """基础长度与关键词检查"""
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

