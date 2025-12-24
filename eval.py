import os
import json
import re
import time
import torch
import bm25s
import logging
import ast
import collections
import matplotlib.pyplot as plt
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
# 1. 工具类与生成器 (保持原样逻辑)
# ==========================================

class GeminiGenerator:
    def __init__(self, model_name, api_key):
        import google.generativeai as genai
        if not api_key:
            raise ValueError("❌ 未检测到 API Key，请设置环境变量 GEMINI_API_KEY")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"🤖 Gemini Generator ({model_name}) 已加载")
        self.max_input_len = 30000 

    def generate(self, input_list, **kwargs):
        responses = []
        for prompt in input_list:
            try:
                if isinstance(prompt, list): prompt = " ".join(prompt)
                clean_prompt = str(prompt)
                result = self.model.generate_content(clean_prompt)
                if result.parts:
                    responses.append(result.text)
                else:
                    responses.append("Error: Empty Response (Safety Block)")
                time.sleep(2) 
            except Exception as e:
                print(f"⚠️ Gemini API Error: {e}")
                time.sleep(5)
                responses.append("Error")
        return responses


from typing import List
from openai import OpenAI

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

# ==========================================
# 2. 评估工具 (Math Logic)
# ==========================================

def extract_math_answer(text):
    """
    (升级版) 从模型输出中提取答案
    逻辑与 _local_extract 保持一致：
    1. 优先找 \boxed{}
    2. 兜底找最后一行
    3. 清洗 '=' 和 '\approx' 以及 LaTeX 杂质
    """
    if not text: return None
    text = str(text).strip()
    
    # 1. 优先提取 \boxed{} 内容
    idx = text.rfind("\\boxed{")
    if idx != -1:
        content_start = idx + 7 
        balance = 0
        for i in range(content_start, len(text)):
            if text[i] == '{': balance += 1
            elif text[i] == '}':
                if balance == 0: return text[content_start:i] 
                balance -= 1
    
    # 2. 兜底策略：取最后一行并清洗
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        if last_line.endswith('.'): last_line = last_line[:-1]
        
        # 清洗 LaTeX 符号
        last_line = last_line.replace('$', '').replace('`', '')
        
        # 去掉 "The Answer is" 前缀
        last_line = re.sub(r'^(The )?Answer( is)?:?', '', last_line, flags=re.IGNORECASE).strip()
        
        # 处理等式 (取等号右边)
        if '=' in last_line: last_line = last_line.split('=')[-1].strip()
        
        # 处理近似符号
        if '\\approx' in last_line: last_line = last_line.split('\\approx')[-1].strip()
        
        # 长度放宽到 100 (原版是 50)
        if len(last_line) < 100: return last_line
        
    return None

def normalize_latex(s):
    """
    (升级版) 标准化 LaTeX 字符串
    逻辑与 _local_norm 保持一致：
    1. 移除 left/right/mathrm 等修饰符
    2. 统一分号、百分号
    3. 再次处理可能残留的 '=' 或 '\in'
    """
    if not s: return ""
    # 基础清洗
    s = str(s).replace('$', '').replace('`', '').replace('\\%', '%')
    s = s.replace("\\dfrac", "\\frac").replace("\\text", "")
    
    # 移除修饰符 (这是关键差异，防止 \left( \right) 导致误判)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\mathrm", "")
    
    # 去除空白
    s = "".join(s.split())
    
    # 再次确保取等号右边 (双重保险)
    if '=' in s: s = s.split('=')[-1]
    if '\\in' in s: s = s.split('\\in')[-1]
    
    return s.rstrip('.').strip()

def _get_field(item, attr: str, key: str = None, default=None):
    """兼容 FlashRAG Dataset 对象 & dict"""
    if hasattr(item, attr):
        return getattr(item, attr)
    if isinstance(item, dict):
        return item.get(key or attr, default)
    return default

def judge_math_item(item):
    """
    返回: (is_right, gold_val, pred_val)
    gold_val/pred_val 是“提取后的原始答案”（未 norm），方便日志/调试。
    """
    pred = _get_field(item, "pred")
    golden_answers = _get_field(item, "golden_answers")
    gold_raw = golden_answers[0] if isinstance(golden_answers, (list, tuple)) and golden_answers else golden_answers

    gold_val = extract_math_answer(gold_raw)
    if gold_val is None:
        gold_val = str(gold_raw).strip() if gold_raw is not None else None

    pred_val = extract_math_answer(pred)

    if not gold_val or not pred_val:
        return False, gold_val, pred_val

    is_right = normalize_latex(gold_val) == normalize_latex(pred_val)
    return is_right, gold_val, pred_val

def evaluate_results(results, experiment_name, result_log_file):
    correct = 0
    total = len(results)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(result_log_file), exist_ok=True)

    with open(result_log_file, "a", encoding="utf-8") as f:
        header = f"\n{'='*20} {experiment_name} {'='*20}\n"
        print(header.strip())
        f.write(header)
        
        for i, item in enumerate(results):
            # 兼容 FlashRAG Dataset 对象和 dict
            pred = item.pred if hasattr(item, 'pred') else item['pred']
            gold_raw = item.golden_answers[0] if hasattr(item, 'golden_answers') else item['golden_answers'][0]
            question = item.question if hasattr(item, 'question') else item['question']

            # 使用升级后的提取逻辑
            is_right, gold_val, pred_val = judge_math_item(item)

            if is_right: correct += 1

            log_entry = (
                f"\n[ID]: {i}\n"
                f"[Question]: {str(question)[:100]}...\n"
                f"[Gold Raw]: ... => [Extracted]: {gold_val}\n"
                f"[Pred Raw]: ...{str(pred)[-50:].replace(chr(10), ' ')} => [Extracted]: {pred_val}\n"
                f"[Result]: {'✅ Correct' if is_right else '❌ Wrong'}\n"
                f"{'-'*30}\n"
            )
            f.write(log_entry)
            if i < 5: print(log_entry.strip()) # 减少一点控制台输出

        acc = correct / total * 100
        summary = (
            f"\n📊 统计 ({experiment_name}):\n"
            f"Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}%\n"
            f"{'='*50}\n"
        )
        print(summary)
        f.write(summary)
    return acc

# ==========================================
# 3. 核心功能函数 (Hydra 适配)
# ==========================================

def prepare_data(cfg: DictConfig, corpus_file: str, test_file: str):
    """
    准备数据：
    1. 检查 corpus_file (优化后的记忆库) 是否存在 (必须存在！)
    2. 生成 test.jsonl (测试集)
    """
    # 🔥 修改 1: 严格检查记忆库是否存在，不存在则报错，绝不自己生成
    if not os.path.exists(corpus_file):
        print(f"❌ 严重错误: 找不到优化后的记忆库文件: {corpus_file}")
        print("   请先运行 optimizer.py 生成该文件！")
        return False
    else:
        print(f"✅ [Memory] 检测到优化后的记忆库: {corpus_file}")

    dataset_name = cfg.experiment.dataset_name
    dataset_config = cfg.experiment.dataset_config
    
    print(f"📥 [Test] 正在加载数据集以构建测试集: {dataset_name}...")
    try:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config)
        else:
            dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

    q_col = cfg.experiment.field_map.question
    a_col = cfg.experiment.field_map.answer
    split_test = "test"
    
    print(f"🔨 [Test] 正在处理测试集...")
    with open(test_file, "w", encoding="utf-8") as f:
        # ... (这里保留原有的测试集切片逻辑，不用动) ...
        # (即原代码中 "if split_test not in dataset:" 开始到 "f.write..." 结束的部分)
        # 为了节省篇幅，这里略过中间未修改的切片逻辑，请保留你原代码中 202-208 行的逻辑
        if split_test not in dataset:
             print(f"❌ 错误: 数据集没有 {split_test} 划分！")
             return False
             
        test_data = dataset[split_test]
        start_idx = int(cfg.experiment.get("start_index", 0) or 0)
        debug_num = cfg.experiment.get("debug_num")
        total_len = len(test_data)
        print(f'\n debug_num是{debug_num}\n')
        if debug_num:
            limit = int(debug_num)
            end_idx = min(start_idx + limit, total_len)
            print(f'\n debug_num是{debug_num}\n')
            print(f'\n end_idx是{end_idx}\n')
        else:
            end_idx = total_len

        if start_idx >= total_len:
            test_data = test_data.select([])
        else:
            indices = range(start_idx, end_idx)
            test_data = test_data.select(indices)

        for i, item in enumerate(test_data):
            q_text = item.get(q_col, "")
            raw_ans = item.get(a_col, "")
            real_id = start_idx + i
            f.write(json.dumps({
                "id": str(real_id),
                "question": q_text,
                "golden_answers": [str(raw_ans)]
            }) + "\n")
            
    return True

def build_index(corpus_file: str, index_dir: str):
    """构建 BM25 索引"""

    print(f"🔨 [Index] 正在为 {corpus_file} 构建 BM25 索引...")
    corpus_texts = []
    # 使用 bm25s 库
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            corpus_texts.append(json.loads(line)['contents'])
    
    corpus_tokens = bm25s.tokenize(corpus_texts)
    retriever_builder = bm25s.BM25()
    retriever_builder.index(corpus_tokens)
    retriever_builder.save(index_dir)
    
    # FlashRAG 要求的额外文件
    with open(os.path.join(index_dir, "stopwords.tokenizer.json"), "w") as f:
        json.dump([], f)
    with open(os.path.join(index_dir, "vocab.tokenizer.json"), "w") as f:
        vocab = corpus_tokens.vocab
        # 兼容性处理
        json.dump({"word_to_id": vocab, "stem_to_sid": vocab, "word_to_stem": {k: k for k in vocab}}, f)
    print("✅ 索引构建完成！")

def analyze_memory_usage(rag_results, cfg: DictConfig, corpus_file: str, vis_image_file: str):
    """
    记忆热度/效用统计与导出 (强化学习版)
    逻辑：
    - 检索命中 & 题目做对: freq += 2 (奖励)
    - 检索命中 & 题目做错: freq -= 2 (惩罚)
    """
    # 这里的 freq_file 从 config 中读取
    freq_file = cfg.paths.eval_freq_file
    
    print("\n🔍 [Analysis] 正在进行全量记忆效用评分 (RL Scoring)...")
    
    all_memory_ids = set()
    id_to_content = {} 
    print(corpus_file)
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                mid = str(item['id'])
                all_memory_ids.add(mid)
                id_to_content[mid] = item.get("contents", "")
    
    except Exception as e:
        print(f"⚠️ 无法读取记忆库文件 {corpus_file}，错误: {e}")
    # # ======== 【jychen】 查看记忆库里的第一条 ========
    # if id_to_content:
    #     first_mid = next(iter(id_to_content))
    #     print(f"\n👀 [DEBUG] 记忆库首条内容检查 (ID: {first_mid}):")
    #     print(f"{id_to_content[first_mid]}")
    #     print("-" * 50)
    # # ==================================================

    # 初始化分数，默认 0
    # 注意：这里我们用 score 代替原来的单纯计数，但变量名在输出时依然叫 freq 以兼容后续脚本
    memory_scores = {mid: 0 for mid in all_memory_ids}
    
    # 统计命中并打分
    total_questions = len(rag_results)
    correct_count = 0

    for item in tqdm(rag_results, desc="Scoring Memories"):
        is_correct, _, _ = judge_math_item(item)
        if is_correct:
            correct_count += 1

        scoreget = cfg.experiment.reward
        scoreloss = cfg.experiment.punishment
        reward = scoreget if is_correct else scoreloss

        retrieved_docs = getattr(item, 'retrieval_result', [])
        # # ======== 【jychen】 查看当前题目检索到的第一条 ========
        # if retrieved_docs:
        #     first_doc = retrieved_docs[0]
        #     # 兼容字典或对象读取 ID
        #     f_id = str(first_doc.get('id')) if isinstance(first_doc, dict) else str(getattr(first_doc, 'id', ''))
            
        #     print(f"\n👀 [DEBUG] 当前题目检索到的 Top1 记忆 (ID: {f_id}):")
        #     # 从之前加载的 id_to_content 中查内容
        #     content = id_to_content.get(f_id, "⚠️ 内容未在 corpus 文件中找到")
        #     print(content)
        #     print("-" * 50)
        # # =======================================================
        for doc in retrieved_docs:
            doc_id = str(doc.get('id')) if isinstance(doc, dict) else str(getattr(doc, 'id', None))
            if doc_id and doc_id in memory_scores:
                memory_scores[doc_id] += reward

    # 排序 (按分数从高到低)
    sorted_memories = sorted(memory_scores.items(), key=lambda x: (-x[1], x[0]))
    
    # 统计信息
    total_mem = len(sorted_memories)
    positive_mem = sum(1 for _, v in sorted_memories if v > 0)
    negative_mem = sum(1 for _, v in sorted_memories if v < 0)
    zero_mem = sum(1 for _, v in sorted_memories if v == 0)
    
    print(f"📊 记忆库评分统计:")
    print(f"   - 总量: {total_mem}")
    print(f"   - 正分(贡献者): {positive_mem} ({(positive_mem/total_mem)*100:.1f}%)")
    print(f"   - 负分(干扰项): {negative_mem} ({(negative_mem/total_mem)*100:.1f}%)")
    print(f"   - 零分(冷门): {zero_mem}")
    print(f"   - 当前题目正确率: {correct_count/total_questions*100:.2f}%")

    # 导出 jsonl (保持 freq 字段名，但存的是分数)
    try:
        print(f"💾 [Save] 正在导出记忆评分结果到: {freq_file}")
        os.makedirs(os.path.dirname(freq_file), exist_ok=True)
        
        with open(freq_file, "w", encoding="utf-8") as f:
            for rank, (mid, score) in enumerate(sorted_memories, start=1):
                record = {
                    "rank": rank,
                    "memory_id": mid,
                    "freq": int(score), # 🔥 这里存的是分数 (-2, 0, 2, 4...)
                    "contents": id_to_content.get(mid, "")
                }
                # print(record["contents"]) #jychen
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("✅ 评分文件导出完成！")
    except Exception as e:
        print(f"❌ 导出失败: {e}")

    # 可视化 (分数分布图)
    if cfg.experiment.visualize_memory:
        print(f"🎨 [Visual] 正在生成分数分布图: {vis_image_file}")
        try:
            ids = [m[0] for m in sorted_memories]
            scores = [m[1] for m in sorted_memories]
            
            display_limit = 30
            if len(ids) > display_limit * 2:
                plot_ids = ids[:display_limit] + ["..."] + ids[-display_limit:]
                plot_scores = scores[:display_limit] + [0] + scores[-display_limit:]
                # 颜色区分：正分蓝，负分红，零分白
                colors = []
                for s in plot_scores:
                    if s > 0: colors.append('skyblue')
                    elif s < 0: colors.append('salmon')
                    else: colors.append('lightgrey')
            else:
                plot_ids = ids
                plot_scores = scores
                colors = ['skyblue' if s > 0 else 'salmon' if s < 0 else 'lightgrey' for s in plot_scores]

            plt.figure(figsize=(15, 6))
            # 画一条 0 分线
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            bars = plt.bar(plot_ids, plot_scores, color=colors, edgecolor='navy')
            plt.title(f'Memory Utility Score (Correct=+2, Wrong=-2)', fontsize=14)
            plt.ylabel('Score')
            plt.xticks(rotation=90, fontsize=8) 
            
            # 显示数值
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if plot_ids[i] != "...": 
                    # 正分显示在条上方，负分显示在条下方
                    y_pos = height if height >= 0 else height - (max(scores)*0.05)
                    va = 'bottom' if height >= 0 else 'top'
                    plt.text(bar.get_x() + bar.get_width()/2., y_pos, f'{int(height)}',
                             ha='center', va=va, fontsize=8)
            
            plt.tight_layout()
            plt.savefig(vis_image_file, dpi=300)
            print("✅ 图片保存成功！")
        except ImportError:
            print("❌ 缺少 matplotlib")
    else:
        print("\n🏆 [Top 10 High-Utility Memories]")
        for mid, score in sorted_memories[:10]:
            print(f"   ID: {mid:<5} | Score: {score}")


# ==========================================
# 4. 主程序 (Hydra Managed)
# ==========================================

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    # 0. 基础设置与路径构造
    print("Visible GPU count:", torch.cuda.device_count())
    
    # 构造文件路径 (全部基于 cfg.paths.root)
    root_dir = cfg.paths.root
    dataset_tag = cfg.experiment.dataset_name.split('/')[-1]
    
    # 定义中间文件路径
    # 🔥 修改这里：文件名加上 _optimized，与 eval.py 逻辑保持一致
    corpus_file = cfg.paths.optimized_memory
    test_file = os.path.join(root_dir, f"{dataset_tag}_test_data.jsonl")
    index_dir = os.path.join(root_dir, f"{dataset_tag}_optimized_bm25_index")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_log_file = os.path.join(root_dir, f"eval_{dataset_tag}_{cfg.model.source}_{cfg.experiment.mode}_{timestamp}.txt")
    vis_image_file = os.path.join(root_dir, f"memory_distribution_{timestamp}.png")

    if os.path.exists(result_log_file): os.remove(result_log_file)
    print(f"📝 结果将保存至: {result_log_file}")
    print(f"🛠️ 模式: {cfg.experiment.mode} | 源: {cfg.model.source} | 数据集: {cfg.experiment.dataset_name}")

    # 1. 数据准备
    if not prepare_data(cfg, corpus_file, test_file): return
    
    # 2. 索引构建 (如果是 rag 或 all 模式)
    if cfg.experiment.mode in ['rag', 'all']:
        build_index(corpus_file, index_dir)
    
    # 3. 初始化 Generator
    generator = None
    config = None # FlashRAG config
    
    model_source = cfg.model.source
    
    if model_source == "gemini":
        print(f"🤖 [Init] 初始化 Gemini: {cfg.model.gemini_name}...")
        api_key = os.environ.get("GEMINI_API_KEY") 
        generator = GeminiGenerator(cfg.model.gemini_name, api_key)
        
        # 构造 FlashRAG 配置字典
        gemini_config_dict = {
            "data_dir": root_dir,
            "save_dir": cfg.paths.rag_cache_dir,
            "device": "cpu",
            "retrieval_method": cfg.experiment.retrieval_method,
            "corpus_path": corpus_file,
            "index_path": index_dir,
            "retriever_model_path": index_dir,
            "generator_model": "huggingface", # 占位
            "generator_model_path": "gpt2",   # 占位
        }
        config = Config(config_dict=gemini_config_dict)

    elif model_source == "sglang":
        print(f"🚀 [Init] 初始化 SGLang Client...")
        
        # 1. 从 config 读取
        sglang_base_url = cfg.model.get("sglang_api_url", "http://127.0.0.1:30000/v1")
        # ⚠️ 确保这里读到的是 "Qwen/Qwen3-4B-Instruct-2507"
        sglang_model_name = cfg.model.get("sglang_model_name", "Qwen/Qwen3-4B-Instruct-2507")
        
        # 2. 构造 FlashRAG 配置字典
        sglang_config_dict = {
            "data_dir": root_dir,
            "save_dir": cfg.paths.rag_cache_dir,
            "corpus_path": corpus_file,
            "index_path": index_dir,
            "retriever_model_path": index_dir,
            "retrieval_method": cfg.experiment.retrieval_method,
            
            # --- 关键修改 ---
            "device": "cpu",
            "gpu_num": 0,
            
            # 1. 告诉 FlashRAG 我们在用类似 OpenAI 的生成协议 (虽然我们实际上是用自定义 Generator 覆盖了它)
            "generator_model": "openai",   
            
            # 2. 🔥🔥🔥 核心修复在这里 🔥🔥🔥
            # 不要让它去加载 "openai" 的 config，而是去加载 Qwen 的 config！
            # PromptTemplate 需要这个路径来下载 tokenizer config
            "generator_model_path": sglang_model_name, 
            
            "generation_method": "openai", 
            "batch_size": cfg.model.batch_size,
            "max_input_len": cfg.model.max_input_len,
            "max_new_tokens": cfg.model.max_new_tokens,
        }
        
        config = Config(config_dict=sglang_config_dict)

        # 4. 初始化 Generator
        generator = SGLangGenerator(
            base_url=sglang_base_url,
            model_name=sglang_model_name,
            max_new_tokens=cfg.model.max_new_tokens,
            batch_size=cfg.model.batch_size,
            temperature=0.7, # 如果需要，这个也可以提到 yaml 里
        )
        print(f"✅ SGLang Generator ({sglang_model_name}) 已连接至 {sglang_base_url}")

    elif model_source == "huggingface":
        hf_name = cfg.model.hf_name
        print(f"📥 [Init] 检查/下载 HF 模型: {hf_name}...")
        try:
            model_path = snapshot_download(repo_id=hf_name)
        except:
            print("❌ 模型下载失败")
            return

        hf_config_dict = {
            "data_dir": root_dir,
            "save_dir": cfg.paths.rag_cache_dir,
            "device": cfg.model.device,
            "gpu_num": torch.cuda.device_count(),
            "generator_model": "huggingface",
            "generator_model_path": model_path,
            "generation_method": "huggingface",
            "batch_size": cfg.model.batch_size,
            "max_input_len": cfg.model.max_input_len,
            "max_new_tokens": cfg.model.max_new_tokens,
        }
        print("🚀 加载 HF 生成器...")
        config = Config(config_dict=hf_config_dict)
        generator = get_generator(config)
        
        # 🔥 Tokenizer 修正 (保持你原有的 padding 修复)
        if hasattr(generator, 'tokenizer'):
            generator.tokenizer.padding_side = 'left' 
            if generator.tokenizer.pad_token is None:
                generator.tokenizer.pad_token = generator.tokenizer.eos_token
                generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id
            generator.tokenizer.model_max_length = cfg.model.max_input_len
        
        if hasattr(generator, 'model'):
            if hasattr(generator.model.config, 'pad_token_id') and generator.model.config.pad_token_id is None:
                generator.model.config.pad_token_id = generator.tokenizer.pad_token_id
        print(f"✅ Tokenizer 修正完成")
    
    else:
        print(f"❌ 未支持的 MODEL_SOURCE: {model_source}")
        return

    # 读取测试数据
    with open(test_file, "r") as f:
        test_dataset_raw = [json.loads(line) for line in f]

    acc_baseline = 0
    acc_rag = 0

    # 格式化 Prompt 辅助函数
    def format_base_prompt(system_text, user_text):
        if model_source == "gemini":
            return f"{system_text}\n\n{user_text}" if system_text else user_text
        prompt = ""
        if system_text: prompt += f"{system_text}\n\n"
        prompt += f"### Question:\n{user_text}\n\n### Answer:\nLet's think step by step."
        return prompt

    # --- Task A: Baseline ---
    if cfg.experiment.mode in ['baseline']:
        print("\n⚔️ [Task A] 正在运行 Baseline ...")
        
        baseline_inputs = []
        for item in test_dataset_raw:
            sys_msg = "You are a math expert. Solve the problem in a brief. Don't answer more than 50 words.End your answer with \\boxed{number}."
            formatted_prompt = format_base_prompt(sys_msg, item['question'])
            baseline_inputs.append(formatted_prompt)

        baseline_preds = generator.generate(baseline_inputs)
        
        baseline_results = []
        for item, pred in zip(test_dataset_raw, baseline_preds):
            baseline_results.append({
                "question": item['question'],
                "golden_answers": item['golden_answers'],
                "pred": pred
            })
        acc_baseline = evaluate_results(baseline_results, "Baseline (No RAG)", result_log_file)

    # --- Task B: FlashRAG ---
    if cfg.experiment.mode in ['rag', 'all']:
        print("\n⚔️ [Task B] 正在运行 FlashRAG (Few-shot Retrieval)...")
        
        # 准备 RAG 配置
        rag_config_dict = OmegaConf.to_container(cfg, resolve=True) # 仅仅是为了获取一些基础类型
        # 将 FlashRAG 需要的特定字段覆盖进去
        rag_update = {
            "data_dir": root_dir,
            "save_dir": cfg.paths.rag_cache_dir,
            "retrieval_method": cfg.experiment.retrieval_method,
            "corpus_path": corpus_file,
            "index_path": index_dir,
            "retriever_model_path": index_dir,
            "topk": cfg.experiment.retrieval_topk,
            # Generator 配置继承之前的
            "device": cfg.model.device,
            "generator_model_path": config['generator_model_path'] if 'generator_model_path' in config else "gpt2"
        }
        
        # 重新实例化 Config 以确保 Retriever 能读到正确参数
        rag_config = Config(config_dict=rag_update)
        retriever = get_retriever(rag_config)
        
        rag_system_part = cfg.experiment.prompts.rag_system
        
        prompt_tpl = PromptTemplate(rag_config, system_prompt=rag_system_part, user_prompt="")
        pipeline = SequentialPipeline(rag_config, prompt_template=prompt_tpl, retriever=retriever, generator=generator)
        dataset_obj = Dataset(rag_config, test_file)
        
        rag_results = pipeline.run(dataset_obj)
        
        acc_rag = evaluate_results(rag_results, f"FlashRAG ({dataset_tag} Memory)", result_log_file)
        
        # 统计记忆热度 (传入 cfg)
        analyze_memory_usage(rag_results, cfg, corpus_file, vis_image_file)

    # --- Summary ---
    if cfg.experiment.mode == 'all':
        summary = (
            f"\n{'='*20} 最终对比结果 {'='*20}\n"
            f"📊 数据集: {cfg.experiment.dataset_name}\n"
            f"🤖 模型: {model_source}\n"
            f"📉 Baseline: {acc_baseline:.2f}%\n"
            f"📈 FlashRAG: {acc_rag:.2f}%\n"
            f"🚀 提升: {acc_rag - acc_baseline:+.2f}%\n"
            f"{'='*50}\n"
        )
        print(summary)
        with open(result_log_file, "a", encoding="utf-8") as f:
            f.write(summary)

if __name__ == "__main__":
    main()