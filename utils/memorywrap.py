import re

def parse_memory(output_text):
    """
    从学生模型的输出中提取 \memory{...} 内容。
    如果没找到 tag，则启用兜底策略（返回全文或尝试清理）。
    """
    if not output_text:
        return ""
        
    # 1. 尝试匹配 \memory{...}，使用 DOTALL 匹配换行
    match = re.search(r'\\memory\{(.*?)\}', output_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 2. 兜底策略：如果模型忘了写 tag，但输出内容看起来还行
    # 这里可以根据情况决定：是返回全文，还是返回空让流程失败
    # 建议返回全文，并在日志里报个 Warning
    print(f"⚠️ Warning: Student output format mismatch (No \\memory tag). Using full text.")
    
    # 简单的清理：去掉可能的 "Sure, ..." 前缀
    clean_text = re.sub(r'^(Sure|Here|Okay).*?:\n', '', output_text, flags=re.IGNORECASE).strip()
    return clean_text