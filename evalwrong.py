import os
import json
import re
import time
import torch
import bm25s
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import snapshot_download

# Hydra & OmegaConf
import hydra
from omegaconf import DictConfig, OmegaConf

# FlashRAG
from flashrag.config import Config
from flashrag.pipeline import SequentialPipeline
from flashrag.utils import get_retriever, get_generator, Dataset
from flashrag.prompt import PromptTemplate

# 屏蔽 transformers 的冗余警告
import transformers
transformers.logging.set_verbosity_error()

# ==========================================
# 1. 核心工具类 (Generator & Eval)
# ==========================================

# ... (此处直接粘贴之前 SGLangGenerator, GeminiGenerator 的代码) ...
from typing import List
from openai import OpenAI

class SGLangGenerator:
    def __init__(self, base_url, model_name, max_new_tokens=1024, batch_size=8, temperature=0.0):
        self.client = OpenAI(api_key="EMPTY", base_url=base_url.rstrip("/"))
        self.model = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_input_len = 4096

    def generate(self, prompts: List[str]) -> List[str]:
        outputs = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            for p in batch:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": p}],
                        temperature=self.temperature,
                        max_tokens=self.max_new_tokens,
                    )
                    outputs.append(resp.choices[0].message.content)
                except Exception as e:
                    print(f"❌ SGLang Error: {e}")
                    outputs.append("")
        return outputs

class GeminiGenerator:
    def __init__(self, model_name, api_key):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, input_list, **kwargs):
        responses = []
        for prompt in input_list:
            try:
                if isinstance(prompt, list): prompt = " ".join(prompt)
                result = self.model.generate_content(str(prompt))
                responses.append(result.text if result.parts else "")
                time.sleep(1) 
            except:
                responses.append("Error")
        return responses

# 🔥 必须使用修复后的判题函数
def extract_math_answer(text):
    if not text: return None
    text = str(text).strip()
    idx = text.rfind("\\boxed{")
    if idx != -1:
        content_start = idx + 7 
        balance = 0
        for i in range(content_start, len(text)):
            if text[i] == '{': balance += 1
            elif text[i] == '}':
                if balance == 0: return text[content_start:i] 
                balance -= 1
    
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        if last_line.endswith('.'): last_line = last_line[:-1]
        last_line = last_line.replace('$', '').replace('`', '')
        last_line = re.sub(r'^(The )?Answer( is)?:?', '', last_line, flags=re.IGNORECASE).strip()
        if '=' in last_line: last_line = last_line.split('=')[-1].strip()
        if '\\approx' in last_line: last_line = last_line.split('\\approx')[-1].strip()
        if len(last_line) < 100: return last_line
    return None

def normalize_latex(s):
    if not s: return ""
    s = str(s).replace('$', '').replace('`', '').replace('\\%', '%')
    s = s.replace("\\dfrac", "\\frac").replace("\\text", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\mathrm", "")
    s = "".join(s.split())
    if '=' in s: s = s.split('=')[-1]
    if '\\in' in s: s = s.split('\\in')[-1]
    return s.rstrip('.').strip()

def evaluate_results(results):
    correct = 0
    total = len(results)
    for item in results:
        pred = item.pred if hasattr(item, 'pred') else item['pred']
        gold = item.golden_answers[0] if hasattr(item, 'golden_answers') else item['golden_answers'][0]
        
        gold_val = extract_math_answer(gold) or str(gold).strip()
        pred_val = extract_math_answer(pred)
        
        if gold_val and pred_val:
            if normalize_latex(gold_val) == normalize_latex(pred_val):
                correct += 1
    return (correct / total * 100) if total > 0 else 0

# ==========================================
# 2. 辅助数据处理函数
# ==========================================

def build_index_if_needed(corpus_path, index_path):
    if os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "vocab.tokenizer.json")):
        print(f"✅ 索引已存在: {index_path}")
        return
    print(f"🔨 正在构建索引: {corpus_path} -> {index_path}")
    
    texts = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(json.loads(line)['contents'])
    
    corpus_tokens = bm25s.tokenize(texts)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(index_path)
    
    # 补充 FlashRAG 需要的文件
    with open(os.path.join(index_path, "stopwords.tokenizer.json"), "w") as f: json.dump([], f)
    with open(os.path.join(index_path, "vocab.tokenizer.json"), "w") as f:
        v = corpus_tokens.vocab
        json.dump({"word_to_id": v, "stem_to_sid": v, "word_to_stem": {k:k for k in v}}, f)

def convert_optimized_memory_to_corpus(memory_file, corpus_file):
    print(f"🔨 正在转换优化记忆: {memory_file} -> {corpus_file}")
    if not os.path.exists(memory_file):
        raise FileNotFoundError(f"找不到优化记忆文件: {memory_file}")
        
    with open(memory_file, 'r') as fin, open(corpus_file, 'w') as fout:
        for line in fin:
            try:
                item = json.loads(line)
                # 优先取优化过的 question
                content = item.get("question") or item.get("contents", "")
                fout.write(json.dumps({"id": str(item['id']), "contents": content}) + "\n")
            except: continue

# ==========================================
# 3. RAG 任务执行器
# ==========================================

def run_rag_task(task_name, cfg, generator, corpus_path, index_path, test_file):
    print(f"\n{'='*20} 正在执行: {task_name} {'='*20}")
    
    # 1. 确保索引存在
    build_index_if_needed(corpus_path, index_path)
    
    # 2. 构造 Config
    rag_update = {
        "data_dir": cfg.paths.root,
        "save_dir": cfg.paths.rag_cache_dir,
        "retrieval_method": cfg.experiment.retrieval_method,
        "corpus_path": corpus_path,
        "index_path": index_path,
        "retriever_model_path": index_path,
        "topk": cfg.experiment.retrieval_topk,
        "device": cfg.model.device,
        "generator_model": "openai", # 占位
        "generator_model_path": "gpt2" # 占位
    }
    rag_config = Config(config_dict=rag_update)
    
    # 3. 初始化 Pipeline
    retriever = get_retriever(rag_config)
    prompt_tpl = PromptTemplate(rag_config, system_prompt=cfg.experiment.prompts.rag_system, user_prompt="")
    pipeline = SequentialPipeline(rag_config, prompt_template=prompt_tpl, retriever=retriever, generator=generator)
    
    # 4. 加载错题集
    dataset = Dataset(rag_config, test_file)
    
    # 5. 运行
    results = pipeline.run(dataset)
    acc = evaluate_results(results)
    print(f"📊 {task_name} 正确率: {acc:.2f}%")
    return acc

# ==========================================
# 4. 主程序
# ==========================================

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    mode = cfg.eval_compare.mode
    wrong_file = cfg.eval_compare.wrong_file
    
    print(f"🚀 启动错题集对比评测 | 模式: {mode}")
    
    if not os.path.exists(wrong_file):
        print(f"❌ 错题集文件不存在: {wrong_file}\n请先运行 wrong_filter.py")
        return

    # --- 🔥 新增: Debug 切片逻辑 ---
    wrong_num = cfg.experiment.get("wrong_num")
    if wrong_num:
        try:
            limit = int(wrong_num)
            print(f"🐛 [Debug Mode] 仅截取前 {limit} 道错题进行测试...")
            
            # 读取原文件
            with open(wrong_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if len(lines) > limit:
                # 构造临时文件名 (例如: wrong_questions_debug_5.jsonl)
                wrong_file_debug = wrong_file.replace(".jsonl", f"_debug_{limit}.jsonl")
                
                # 写入切片后的数据
                with open(wrong_file_debug, "w", encoding="utf-8") as f:
                    f.writelines(lines[:limit])
                
                # 指针重定向：让后续流程读这个临时文件
                wrong_file = wrong_file_debug
                print(f"   已生成临时切片文件: {wrong_file}")
            else:
                print(f"   错题总数 ({len(lines)}) 少于 wrong_num ({limit})，无需切片。")
        except Exception as e:
            print(f"⚠️ Debug 切片失败: {e}，将使用全量数据。")
    # -----------------------------

    print(f"📂 当前使用的测试文件: {wrong_file}")

    # --- 初始化 Generator (只初始化一次，复用) ---
    model_source = cfg.model.source
    generator = None
    if model_source == "sglang":
        url = cfg.model.get("sglang_api_url", "http://127.0.0.1:30000/v1")
        name = cfg.model.get("sglang_model_name", "Qwen/Qwen3-4B-Instruct-2507")
        generator = SGLangGenerator(url, name, batch_size=cfg.model.batch_size)
        print("✅ SGLang Generator Ready")
    elif model_source == "gemini":
        generator = GeminiGenerator(cfg.model.gemini_name, os.environ.get("GEMINI_API_KEY"))
        print("✅ Gemini Generator Ready")
    elif model_source == "huggingface":
        # 如果需要支持 HF，可以在这里加，但建议错题本用 SGLang 跑得快
        print("⚠️ 建议使用 SGLang 进行错题本快速验证")
        return

    results_summary = {}

    # --- Task 1: 原始记忆库 (Original) ---
    if mode in ["original", "both"]:
        orig_cfg = cfg.eval_compare.original
        if not os.path.exists(orig_cfg.corpus_path):
            print(f"⚠️ 原始语料不存在: {orig_cfg.corpus_path} (请先跑 pre.py)")
        else:
            acc = run_rag_task(
                "原始记忆 (Original)", cfg, generator,
                orig_cfg.corpus_path, orig_cfg.index_path, wrong_file
            )
            results_summary["Original"] = acc

    # --- Task 2: 优化记忆库 (Optimized) ---
    if mode in ["optimized", "both"]:
        opt_cfg = cfg.eval_compare.optimized
        # 实时转换 (保证最新)
        convert_optimized_memory_to_corpus(opt_cfg.memory_file, opt_cfg.corpus_path)
        
        acc = run_rag_task(
            "优化记忆 (Optimized)", cfg, generator,
            opt_cfg.corpus_path, opt_cfg.index_path, wrong_file
        )
        results_summary["Optimized"] = acc

    # --- 最终对比报告 ---
    print("\n" + "="*40)
    print("🏆 错题集复盘对比报告")
    if wrong_num: print(f"(Debug: Top {wrong_num})")
    print("="*40)
    print(f"{'策略':<20} | {'正确率':<10}")
    print("-" * 35)
    
    base_acc = results_summary.get("Original", 0)
    opt_acc = results_summary.get("Optimized", 0)
    
    if "Original" in results_summary:
        print(f"{'Original RAG':<20} | {base_acc:.2f}%")
    if "Optimized" in results_summary:
        print(f"{'Optimized RAG':<20} | {opt_acc:.2f}%")
    
    if mode == "both":
        diff = opt_acc - base_acc
        icon = "🚀" if diff > 0 else "📉"
        print("-" * 35)
        print(f"效果提升: {icon} {diff:+.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()