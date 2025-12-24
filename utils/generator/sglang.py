from typing import List
from openai import OpenAI
import os
class SGLangGenerator:
    """一个最小实现的生成器，适配 FlashRAG 的 generator.generate(prompts) 接口。"""
    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_new_tokens: int = 1024,
        batch_size: int = 8,
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
        outputs: List[str] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            for p in batch:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": p}],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
                outputs.append(resp.choices[0].message.content)
        return outputs
