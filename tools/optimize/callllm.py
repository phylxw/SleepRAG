from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import DictConfig
from typing import List
import os
import torch
from utils.toolfunction import clean_special_chars


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

def call_llm(prompt: str, cfg: DictConfig, max_new_tokens: int = None) -> str:
    """统一的大模型调用接口，单条调用"""
    model_source = cfg.model.optimize
    # 如果没传 max_new_tokens，就用 config 里的默认值
    if max_new_tokens is None:
        max_new_tokens = cfg.model.max_new_tokens

    # --- Gemini ---
    if model_source == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            return "Skipped (No GEMINI_API_KEY)"
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel(cfg.model.gemini_name)
            print("  🤖 [Gemini] 正在生成...", end="", flush=True)
            resp = model.generate_content(prompt)
            print(" 完成")
            return clean_special_chars(resp.text.strip())
        except Exception as e:
            print(f"\n❌ [Gemini Error]: {e}")
            return ""

    # --- HuggingFace 本地 ---
    elif model_source == "huggingface":
        if GLOBAL_MODEL is None:
            print("⚠️ [Local] LLM 尚未初始化")
            return ""

        try:
            print("  🚀 [Local] 正在生成...", end="", flush=True)
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
            print(" 完成")
            return clean_special_chars(response.strip())
        except Exception as e:
            print(f"\n❌ [Local Error]: {e}")
            return ""

    # --- SGLang ---
    elif model_source == "sglang":
        if GLOBAL_SGLANG_CLIENT is None:
            return "Skipped (Client Not Initialized)"
        
        model_name = cfg.model.get("sglang_model_name", "Qwen/Qwen3-4B-Instruct-2507")
        try:
            print("  🚀 [SGLang] 正在推理...", end="", flush=True)
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
            print(" 完成")
            return clean_special_chars(content.strip())
        except Exception as e:
            print(f"\n❌ [SGLang Error]: {e}")
            return ""

    return ""

def call_llm_batch(prompts: List[str], cfg: DictConfig, max_new_tokens: int = None) -> List[str]:
    """批量调用 LLM"""
    if not prompts:
        return []
    
    model_source = cfg.model.optimize
    if max_new_tokens is None:
        max_new_tokens = cfg.model.max_new_tokens

    # Gemini：简单循环
    if model_source == "gemini":
        results = []
        for p in prompts:
            results.append(call_llm(p, cfg, max_new_tokens=max_new_tokens))
        return results

    # SGLang: 简单循环调用 (Server端会自动处理并发)
    if model_source == "sglang":
        results = []
        # 虽然这里写的是循环，但 SGLang Server 的吞吐很高，速度通常比本地 HF Batch 快
        # 如果需要极致并发，可以使用 asyncio 或 ThreadPoolExecutor，但简单循环通常足够快且稳定
        for p in prompts:
            results.append(call_llm(p, cfg, max_new_tokens=max_new_tokens))
        return results

    # HuggingFace 本地
    if model_source == "huggingface":
        if GLOBAL_MODEL is None:
            print("⚠️ [Local] LLM 尚未初始化")
            return [""] * len(prompts)

        try:
            print(f"  🚀 [Local-Batch] 正在批量生成 {len(prompts)} 条...", end="", flush=True)
            
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

            # 批量 Tokenize + Padding
            model_inputs = GLOBAL_TOKENIZER(
                text_list,
                return_tensors="pt",
                padding=True, # 关键
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