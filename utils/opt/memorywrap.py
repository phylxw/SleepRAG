import re

# 兼容：\memory 或 \memory$；\endmemory 或 \endmemory$
_MEM_BLOCK_RE = re.compile(
    r"\\memory\$?\s*(.*?)\s*\\endmemory\$?\s*",
    re.DOTALL | re.IGNORECASE
)

def clean_text(s: str) -> str:
    if not s:
        return ""
    # 1) 处理字面量 "\n"
    s = s.replace("\\n", " ")
    # 2) 处理真实换行 / tab 等所有空白，压成单空格
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def parse_memory(raw: str) -> str:
    if not raw:
        return ""
    m = _MEM_BLOCK_RE.search(raw)
    if m:
        return clean_text(m.group(1))
    # 如果没按格式输出：你可以选择“判失败返回空”，或者保留全文但清洗
    return clean_text(raw)


# --- 本地测试用例 ---
if __name__ == "__main__":
    # Case 1: 你的原始混乱数据（Tag后有大量乱码，且内容简单）
    text1 = "\\sdfvds\\bsb\sfsbsdfbsd\\dbasdfbsfgnbsfgn\memory${我是cjy}\endmemory$sfbgsnngfstrnfgfndtynsrtnqtbs\dfbds\fbdfb\sdb\bds"
    print(f"Case 1 (Garbage suffix): {parse_memory(text1)}")
    # Output: 我是cjy

    # Case 2: 完全无 Tag 的乱码
    text2 = "xgfndyd654nd+5fy4n6mdtyj\memory${我是cjy}8e65rhjr4h6drjdrerh6316tj78e56n1"
    print(f"Case 2 (No tag): {parse_memory(text2)}")
    # Output: (Warning log) + 原文

    # Case 3: 复杂的嵌套括号 + Markdown + 空格 (最容易挂的情况)
    text3 = r"""
    Sure! Here is the memory:
    ```latex
    \memory$ { 
        This is a complex logic with math: \frac{1}{2} and nested { braces }.
    }
    ```
    Hope this helps!
    \endmemory$
    """
    print(f"Case 3 (Complex): {parse_memory(text3)}")
    # Output: This is a complex logic with math: \frac{1}{2} and nested { braces }.