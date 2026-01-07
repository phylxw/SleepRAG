from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import DictConfig
from typing import List
import os
import torch
from utils.toolfunction import clean_special_chars
import logging
from tqdm import tqdm # [新增]
import concurrent.futures
# [新增] 屏蔽 httpx 和 httpcore 的 INFO 日志，防止刷屏
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def init_llm(cfg: DictConfig):
    """初始化 LLM"""
    global GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_SGLANG_CLIENT
    
    model_source = cfg.model.optimize

    if model_source == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            print(f"🤖 [Init] Gemini API ({cfg.model.gemini_name}) 已配置")
        else:
            print("⚠️ [Init] 未检测到 GEMINI_API_KEY，Gemini 相关功能会被跳过")
            
    elif model_source == "huggingface":
        hf_name = cfg.model.hf_name
        print(f"📥 [Init] 正在加载本地模型: {hf_name} ...")
        try:
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
                hf_name,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            ).eval()
            
            # 🔥 [Critical Fix] 批量生成必须设置 left padding
            GLOBAL_TOKENIZER.padding_side = 'left'
            if GLOBAL_TOKENIZER.pad_token is None:
                GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token
                GLOBAL_TOKENIZER.pad_token_id = GLOBAL_TOKENIZER.eos_token_id
            
            print(f"✅ [Init] 本地模型加载完成！(Padding side set to left)")
        except Exception as e:
            print(f"❌ [Init] 本地模型加载失败: {e}")
            print("💡 提示: 请检查 HuggingFace 权限和网络")

    elif model_source == "sglang":
        try:
            from openai import OpenAI
            # 从配置读取 URL，默认本地端口
            api_url = cfg.model.get("sglang_api_url", "http://127.0.0.1:30000/v1")
            api_key = "EMPTY" # SGLang 本地部署不需要真实 Key
            
            GLOBAL_SGLANG_CLIENT = OpenAI(base_url=api_url, api_key=api_key)
            print(f"✅ [Init] SGLang Client 已连接至 {api_url}")
        except ImportError:
            print("❌ [Init] 缺少 openai 库，请运行 `pip install openai`")

def call_llm(prompt: str, cfg: DictConfig, max_new_tokens: int = None, verbose: bool = True) -> str:
    """
    统一的大模型调用接口，单条调用
    新增 verbose 参数：True=打印进度(默认), False=静默模式(用于Batch)
    """
    model_source = cfg.model.optimize
    if max_new_tokens is None:
        max_new_tokens = cfg.model.max_new_tokens

    # --- Gemini ---
    if model_source == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            return "Skipped (No GEMINI_API_KEY)"
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel(cfg.model.gemini_name)
            if verbose:
                print(" 🤖 [Gemini] 正在生成...", end="", flush=True)
            resp = model.generate_content(prompt)
            if verbose:
                print(" 完成")
            return clean_special_chars(resp.text.strip())
        except Exception as e:
            if verbose: print(f"\n❌ [Gemini Error]: {e}")
            return ""

    # --- HuggingFace 本地 ---
    elif model_source == "huggingface":
        if GLOBAL_MODEL is None:
            if verbose: print("⚠️ [Local] LLM 尚未初始化")
            return ""

        try:
            if verbose:
                print(" 🚀 [Local] 正在生成...", end="", flush=True)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = GLOBAL_TOKENIZER.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = GLOBAL_TOKENIZER(
                [text],
                return_tensors="pt",
                truncation=True,
                max_length=cfg.model.max_input_len,
            ).to(GLOBAL_MODEL.device)

            with torch.no_grad():
                generated_ids = GLOBAL_MODEL.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            # 只取新增的部分
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = GLOBAL_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if verbose:
                print(" 完成")
            return clean_special_chars(response.strip())
        except Exception as e:
            if verbose: print(f"\n❌ [Local Error]: {e}")
            return ""

    # --- SGLang ---
    elif model_source == "sglang":
        if GLOBAL_SGLANG_CLIENT is None:
            return "Skipped (Client Not Initialized)"
        
        model_name = cfg.model.get("sglang_model_name", "Qwen/Qwen3-4B-Instruct-2507")
        try:
            if verbose:
                print(" 🚀 [SGLang] 正在推理...", end="", flush=True)
            
            resp = GLOBAL_SGLANG_CLIENT.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=max_new_tokens
            )
            content = resp.choices[0].message.content
            
            if verbose:
                print(" 完成")
            return clean_special_chars(content.strip())
        except Exception as e:
            if verbose: print(f"\n❌ [SGLang Error]: {e}")
            return ""

    return ""


def call_llm_batch(prompts: List[str], cfg: DictConfig, max_new_tokens: int = None) -> List[str]:
    """批量调用 LLM (SGLang 并发优化 + 进度条版)"""
    if not prompts:
        return []
    
    model_source = cfg.model.optimize
    if max_new_tokens is None:
        max_new_tokens = cfg.model.max_new_tokens

    # --- SGLang 并发加速 (带 tqdm 进度条) ---
    if model_source == "sglang":
        max_workers = cfg.model.get("batch_size", 32)
        
        results = [None] * len(prompts)
        
        # 使用线程池并发，并传入 verbose=False 禁止内部 print
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(call_llm, p, cfg, max_new_tokens, verbose=False): i 
                for i, p in enumerate(prompts)
            }
            
            # 🔥 修改点：使用 tqdm 包裹迭代器，显示进度条
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(prompts), desc="🚀 SGLang Batch"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # 只有出错才打印
                    print(f"\n❌ [Batch Error] Task {idx} failed: {e}")
                    results[idx] = ""
        
        return results

    # --- Gemini (保持串行) ---
    if model_source == "gemini":
        results = []
        # Gemini 也可以加个简单的进度条，如果需要的话
        for p in tqdm(prompts, desc="🤖 Gemini Batch"):
            results.append(call_llm(p, cfg, max_new_tokens=max_new_tokens))
        return results

    # --- HuggingFace 本地 (HF本身支持Batch推理，逻辑不变) ---
    if model_source == "huggingface":
        if GLOBAL_MODEL is None:
            print("⚠️ [Local] LLM 尚未初始化")
            return [""] * len(prompts)

        try:
            print(f" 🚀 [Local-Batch] 正在批量生成 {len(prompts)} 条...", end="", flush=True)
            
            messages_list = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": p}
                ]
                for p in prompts
            ]
            text_list = [
                GLOBAL_TOKENIZER.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for msgs in messages_list
            ]

            model_inputs = GLOBAL_TOKENIZER(
                text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.model.max_input_len,
            ).to(GLOBAL_MODEL.device)

            with torch.no_grad():
                generated_ids = GLOBAL_MODEL.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )

            results = []
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
                new_token_ids = output_ids[len(input_ids):]
                text = GLOBAL_TOKENIZER.decode(new_token_ids, skip_special_tokens=True)
                results.append(clean_special_chars(text.strip()))
            print(" 完成")
            return results

        except Exception as e:
            print(f"\n❌ [Local-Batch Error]: {e}")
            return [""] * len(prompts)

    return [""] * len(prompts)