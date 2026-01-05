from typing import Dict, List
import hydra
from omegaconf import DictConfig

def generate_gradient_prompt(content: str, neg_queries: List[str], cfg: DictConfig) -> str:
    """
    🔥 [Step 1: Backward Pass - Expert]
    请求专家模型进行“归因分析”，计算文本梯度。
    """
    neg_text = "\n".join([f"- {q}" for q in neg_queries[:5]])
    raw_prompt = cfg.optimizer.prompts.gradient_generate
    prompt = raw_prompt.format(content=content, neg_text=neg_text)
    return prompt

def apply_gradient_prompt(content: str, gradient: str, good_examples: str, cfg: DictConfig) -> str:
    """
    🔥 [Step 2: Update Step - Student]
    请求 Qwen 根据专家的梯度重写记忆。
    """
    momentum_part = ""
    if good_examples:
        momentum_part = f"\n[Reference (Momentum)]\nHigh-quality neighbors:\n{good_examples}\n"

    # 尝试读取 config 里的模板，否则用默认
    template = cfg.optimizer.prompts.apply_gradient
    return template.format(content=content, gradient=gradient, momentum_part=momentum_part)

def summarize_experience_prompt(target_text: str, good_neighbors: List[str], cfg: DictConfig) -> str:
    """旧逻辑：模仿"""
    good_examples_text = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(good_neighbors))
    template = cfg.optimizer.prompts.expand_low_freq
    prompt = template.format(text=target_text, good_examples=good_examples_text)
    return prompt

def expand_low_freq_memory_prompt(text: str, good_examples: str, cfg: DictConfig) -> str:
    """旧逻辑：自省"""
    template = cfg.optimizer.prompts.expand_low_freq
    prompt = template.format(text=text, good_examples=good_examples)
    return prompt