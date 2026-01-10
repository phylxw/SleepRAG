import re

def parse_memory(output_text):
    """
    终极版解析器：从模型输出中提取 \memory{...} 内容。
    
    ✅ 支持 \memory { ... } (中间有空格)
    ✅ 支持 Markdown 包裹 (```latex ... ```)
    ✅ 支持 嵌套括号 (例如内容里有 LaTeX 公式 \frac{a}{b})
    ✅ 自动去除前缀废话 (Sure, here is...)
    """
    if not output_text:
        return ""

    # 1. 清洗 Markdown 标记 (模型经常喜欢把输出包在代码块里)
    # 去掉 ```latex, ```text, 或纯 ```
    text = re.sub(r"```[a-zA-Z]*", "", output_text).replace("```", "").strip()

    # 2. 定位 \memory{ 的起始位置
    # 使用正则查找开头，允许 \memory 和 { 之间有空格，忽略大小写
    start_pattern = re.compile(r"\\memory\s*\{", re.IGNORECASE)
    match = start_pattern.search(text)

    if not match:
        # --- 兜底策略 ---
        # 如果真的没找到 tag，说明模型完全没遵循格式。
        # 这里保留你的 Warning，并尝试清洗前缀后返回全文。
        print(f"⚠️ Warning: Student output format mismatch (No \\memory tag). Using full text.")
        
        # 去掉常见的 "Sure, ..." "Here is..." 前缀
        clean_text = re.sub(r'^(Sure|Here|Okay|Certainly).*?:\n', '', text, flags=re.IGNORECASE).strip()
        
        # 如果清洗后还是空的，或者原文就没 tag，只能返回清洗后的全文碰运气
        return clean_text

    # 3. 栈式解析 (Brace Counting) - 解决嵌套括号和尾部垃圾字符的关键
    # match.end() - 1 是 '{' 的索引位置
    current_idx = match.end() - 1
    depth = 0
    extracted_chars = []
    
    # 从第一个 '{' 开始遍历
    for char in text[current_idx:]:
        if char == '{':
            depth += 1
            # 只有当深度 > 1 时才记录字符（不记录最外层的 '{'）
            if depth > 1:
                extracted_chars.append(char)
        elif char == '}':
            depth -= 1
            if depth == 0:
                # 深度归零，说明找到了匹配的闭合括号，解析结束
                break
            extracted_chars.append(char)
        else:
            # 记录普通字符
            if depth > 0:
                extracted_chars.append(char)
    
    # 拼接结果并去除首尾空白
    return "".join(extracted_chars).strip()

# --- 本地测试用例 ---
if __name__ == "__main__":
    # Case 1: 你的原始混乱数据（Tag后有大量乱码，且内容简单）
    text1 = "\\sdfvds\\bsb\sfsbsdfbsd\\dbasdfbsfgnbsfgn\memory{我是cjy}sfbgsnngfstrnfgfndtynsrtnqtbs\dfbds\fbdfb\sdb\bds"
    print(f"Case 1 (Garbage suffix): {parse_memory(text1)}")
    # Output: 我是cjy

    # Case 2: 完全无 Tag 的乱码
    text2 = "xgfndyd654nd+5fy4n6mdtyj8e65rhjr4h6drjdrerh6316tj78e56n1"
    print(f"Case 2 (No tag): {parse_memory(text2)}")
    # Output: (Warning log) + 原文

    # Case 3: 复杂的嵌套括号 + Markdown + 空格 (最容易挂的情况)
    text3 = """
    Sure! Here is the memory:
    ```latex
    \memory { 
        This is a complex logic with math: \frac{1}{2} and nested { braces }.
    }
    ```
    Hope this helps!
    """
    print(f"Case 3 (Complex): {parse_memory(text3)}")
    # Output: This is a complex logic with math: \frac{1}{2} and nested { braces }.