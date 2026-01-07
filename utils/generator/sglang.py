from typing import List
from openai import OpenAI
import os
import concurrent.futures
import logging
from tqdm import tqdm  # 记得 pip install tqdm

# 1. 全局或类初始化时屏蔽 httpx 的烦人日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class SGLangGenerator:
    """
    适配 FlashRAG 的 SGLang 生成器 (并发优化 + 进度条版)
    """
    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_new_tokens: int = 1024,
        batch_size: int = 32, 
        temperature: float = 0.0,
    ):
        self.client = OpenAI(
            api_key=os.getenv("SGLANG_API_KEY", "EMPTY"),
            base_url=base_url.rstrip("/"),
        )
        self.model = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_input_len = 4096

    def generate(self, prompts: List[str]) -> List[str]:
        """
        使用线程池并发发送请求，并显示进度条
        """
        if not prompts:
            return []

        # 定义单个请求的发送逻辑
        def _send_request(prompt):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                # 只有出错了才打印，保持清爽
                print(f"❌ SGLang Request Error: {e}")
                return ""

        results = [None] * len(prompts)
        
        # 使用线程池并发执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            future_to_idx = {
                executor.submit(_send_request, p): i 
                for i, p in enumerate(prompts)
            }
            
            # 🔥 核心修改：用 tqdm 包裹 as_completed，实现进度条
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(prompts), desc="🚀 SGLang Inference"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"Task {idx} generated an exception: {exc}")
                    results[idx] = ""

        return results