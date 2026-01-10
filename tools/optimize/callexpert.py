import os
import concurrent.futures
from typing import List
from omegaconf import DictConfig
from utils.toolfunction import clean_special_chars
import logging
from tqdm import tqdm

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

    # 🔥 [修改点 1] 将 qwen 加入到 openai 兼容列表
    elif source in ["openai", "sglang", "qwen"]:
        try:
            from openai import OpenAI
            # 默认配置 (OpenAI)
            base_url = os.environ.get("EXPERT_BASE_URL", "https://api.openai.com/v1")
            api_key = os.environ.get("EXPERT_API_KEY")
            
            # 针对 SGLang 的特殊配置
            if source == "sglang":
                base_url = expert_cfg.get("sglang_api_url", "http://127.0.0.1:30000/v1")
                api_key = "EMPTY"
            
            # 🔥 [修改点 2] 针对 Qwen (DashScope) 的特殊配置
            elif source == "qwen":
                # 阿里云百炼兼容模式 endpoint
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
                # 优先读环境变量 DASHSCOPE_API_KEY，如果没有则读 EXPERT_API_KEY，最后才是硬编码（不推荐）
                api_key = "sk-dab5c76d636e4e4b9567b0c45d73ba83"
                
                # 如果环境变量没设，为了方便调试，你可以暂时用这里的硬编码 (但生产环境请删掉)
                if not api_key:
                    api_key = "sk-dab5c76d636e4e4b9567b0c45d73ba83" # 你的 Key

            GLOBAL_EXPERT_CLIENT = OpenAI(base_url=base_url, api_key=api_key)
            print(f"✅ [Expert-Init] {source.upper()} Client ({expert_cfg.name}) 就绪 | URL: {base_url}")
        except ImportError:
            print("❌ [Expert-Init] 缺少 openai 库")
    else:
        print(f"⚠️ [Expert-Init] 未知的专家源: {source}")


def call_expert(prompt: str, cfg: DictConfig) -> str:
    """单条调用"""
    global GLOBAL_EXPERT_CLIENT
    if GLOBAL_EXPERT_CLIENT is None: return None

    source = cfg.expert_model.source
    model_name = cfg.expert_model.name
    
    try:
        if source == "gemini":
            resp = GLOBAL_EXPERT_CLIENT.generate_content(prompt)
            return clean_special_chars(resp.text.strip())
        
        # 🔥 [修改点 3] Qwen 也走这里，但注意：这里不使用 stream=True
        # 因为在代码逻辑中，我们需要完整的字符串返回，而不是生成器
        elif source in ["openai", "sglang", "qwen"]:
            resp = GLOBAL_EXPERT_CLIENT.chat.completions.create(
                model=model_name, # 这里会传入 qwen-max 或 qwen3-max
                messages=[
                    {"role": "system", "content": "You are a helpful and critical AI expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                # 注意：如果 Qwen 报错 "max_tokens too large"，可以适当调小或注释掉
                # qwen-max 支持长文本，一般没问题
                # max_tokens=1024, 
                stream=False  # ❌ 关掉流式，方便后续处理
            )
            return clean_special_chars(resp.choices[0].message.content.strip())

    except Exception as e:
        print(f"❌ [Expert Error]: {e}")
        return None


# 1. 屏蔽 httpx 和 httpcore 的 INFO 日志，防止刷屏
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def call_expert_batch(prompts: List[str], cfg: DictConfig) -> List[str]:
    """
    🔥 批量并发调用专家模型
    """
    if not prompts: return []
    
    source = cfg.expert_model.source
    
    # 🔥 [修改点 4] 允许 Qwen 进行并发
    if source in ["sglang", "openai", "qwen"]:
        # Qwen 的并发限制：
        # 如果是普通账号，Qwen-max 的并发 (QPS) 可能较低。
        # 如果报错 429 Too Many Requests，请把 max_workers 改小 (例如 2 或 4)
        max_workers = 16 
        
        results = [None] * len(prompts)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(call_expert, p, cfg): i 
                for i, p in enumerate(prompts)
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(prompts), desc=f"🧠 {source.upper()} Batch"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"\n❌ [Expert Batch Error] Task {idx} failed: {e}")
                    results[idx] = ""
                    
        return results

    # Gemini 保持原有逻辑
    results = []
    for p in tqdm(prompts, desc="🤖 Gemini Expert"):
        results.append(call_expert(p, cfg))
    return results