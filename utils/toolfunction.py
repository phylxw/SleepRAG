import torch

def clean_special_chars(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.replace('\u2028', ' ').replace('\u2029', ' ')


def has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False
    
def format_base_prompt(system_text, user_text,model_source):
    """
    修改为符合 Qwen Instruct 训练习惯的格式
    """
    # 如果是 Gemini，保持原样（Gemini API 会自己处理）
    if model_source == "gemini":
        return f"{system_text}\n\n{user_text}" if system_text else user_text
    
    # --- 针对 SGLang / Qwen 的修改 ---
    
    # 方案 A: 如果你的 generator 支持传入 messages 列表 (推荐)
    # 很多 FlashRAG 的 generator 内部会处理 list
    # return [
    #     {"role": "system", "content": system_text},
    #     {"role": "user", "content": user_text}
    # ]

    # 方案 B: 如果 generator 只吃 string，我们手动模拟 ChatML (最稳妥)
    # 这就是你挂载的 Jinja 本该做的事，但在这里我们手动做，确保万无一失
    prompt = ""
    if system_text:
        prompt += f"<|im_start|>system\n{system_text}<|im_end|>\n"
    
    prompt += f"<|im_start|>user\n{user_text}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\nLet's think step by step.\n" # 强制开启 CoT
    
    return prompt