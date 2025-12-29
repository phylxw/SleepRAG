import os
import concurrent.futures
from typing import List
from omegaconf import DictConfig
from utils.toolfunction import clean_special_chars

# 定义一个全局的专家客户端
GLOBAL_EXPERT_CLIENT = None

def init_expert_llm(cfg: DictConfig):
    """初始化专家模型 (Teacher Model)"""
    global GLOBAL_EXPERT_CLIENT
    expert_cfg = cfg.expert_model
    source = expert_cfg.source
    
    print(f"👨‍🏫 [Expert-Init] 正在初始化专家模型: {source} ({expert_cfg.name})...")

    if source == "gemini":
        try:
            import google.generativeai as genai
            api_key = os.environ.get("EXPERT_API_KEY") or os.environ.get("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            GLOBAL_EXPERT_CLIENT = genai.GenerativeModel(expert_cfg.name)
            print(f"✅ [Expert-Init] Gemini ({expert_cfg.name}) 就绪")
        except ImportError:
            print("❌ [Expert-Init] 缺少 google-generativeai 库")

    elif source in ["openai", "sglang"]:
        try:
            from openai import OpenAI
            # SGLang 也是用 OpenAI 客户端
            base_url = os.environ.get("EXPERT_BASE_URL", "https://api.openai.com/v1")
            api_key = os.environ.get("EXPERT_API_KEY")
            
            if source == "sglang":
                base_url = expert_cfg.get("sglang_api_url", "http://127.0.0.1:30000/v1")
                api_key = "EMPTY"

            GLOBAL_EXPERT_CLIENT = OpenAI(base_url=base_url, api_key=api_key)
            print(f"✅ [Expert-Init] {source.upper()} Client ({expert_cfg.name}) 就绪")
        except ImportError:
            print("❌ [Expert-Init] 缺少 openai 库")
    else:
        print(f"⚠️ [Expert-Init] 未知的专家源: {source}")


def call_expert(prompt: str, cfg: DictConfig) -> str:
    """单条调用 (内部逻辑保持不变，供 Batch 调用使用)"""
    global GLOBAL_EXPERT_CLIENT
    if GLOBAL_EXPERT_CLIENT is None: return None

    source = cfg.expert_model.source
    model_name = cfg.expert_model.name
    
    try:
        if source == "gemini":
            resp = GLOBAL_EXPERT_CLIENT.generate_content(prompt)
            return clean_special_chars(resp.text.strip())
        
        elif source in ["openai", "sglang"]:
            resp = GLOBAL_EXPERT_CLIENT.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful and critical AI expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                # SGLang 专家通常需要长一点的输出空间写分析
                max_tokens=1024 
            )
            return clean_special_chars(resp.choices[0].message.content.strip())

    except Exception as e:
        print(f"❌ [Expert Error]: {e}")
        return None

def call_expert_batch(prompts: List[str], cfg: DictConfig) -> List[str]:
    """
    🔥 [New] 批量并发调用专家模型
    对于 SGLang/OpenAI，使用多线程并发发送请求，服务端会自动 Batch 处理。
    """
    if not prompts: return []
    
    source = cfg.expert_model.source
    
    # 1. 如果是 SGLang/OpenAI，使用线程池并发 (这是提速的关键！)
    if source in ["sglang", "openai"]:
        # 并发数可以设大一点，比如 16 或 32，SGLang 处理得过来
        max_workers = 16 
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(call_expert, p, cfg) for p in prompts]
            # 获取结果 (保持顺序)
            results = [f.result() for f in futures]
        return results

    # 2. 如果是 Gemini，考虑到速率限制 (Rate Limit)，建议串行或保守并发
    # 这里保持简单循环，避免 429 Error
    results = []
    for p in prompts:
        results.append(call_expert(p, cfg))
    return results