import os
import json
import re
import time
import torch
import bm25s
import logging
import ast
import collections # 新增: 用于计数
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import snapshot_download
from flashrag.config import Config
from flashrag.pipeline import SequentialPipeline
from flashrag.utils import get_retriever, get_generator, Dataset
from flashrag.prompt import PromptTemplate
import matplotlib.pyplot as plt
import json
# 屏蔽 transformers 的冗余警告
import transformers
transformers.logging.set_verbosity_error()

# ==========================================
# 🛠️ 核心配置区域 (已修改为 GSM8K)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
# 1. 实验控制开关
# 选项: 'baseline' (只测原模型), 'rag' (只测FlashRAG), 'all' (对比测试)
EXPERIMENT_MODE = "all" 

# 🔥 [新增] 记忆热度统计开关
# True: 画出记忆调用频次分布图 (保存为png)
# False: 仅在终端输出频次最高的 Top 30 记忆ID
VISUALIZE_MEMORY_DISTRIBUTION = True

# 2. 调试样本数
# 选项: 10 (快速测试), None (跑全量)
# MATH 测试集有 1319 条，建议先设为 20-50 条跑通流程
DEBUG_NUM = None

# 3. 模型设置
# 选项: 'huggingface' (本地/HF模型) 或 'gemini' (Google API)
MODEL_SOURCE = "huggingface" 

# 选项：Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen2-1.5B-Instruct
# 建议使用 Qwen2.5-7B-Instruct，效果更稳定
HF_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507" 

# [Gemini 配置]
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 

# ==========================================
# 4. 数据集配置 (MATH / LightEval 专用配置)
# ==========================================
# lighteval/MATH 其实通常指向 hendrycks/competition_math
# 建议直接使用原始源，或者确保你引用的库存在
DATASET_NAME = "DigitalLearningGmbH/MATH-lighteval" 
# MATH 数据集需要指定子集，"all" 表示加载代数、几何等所有类别
DATASET_CONFIG = "algebra" 

SPLIT_TRAIN = "train" 
SPLIT_TEST = "test"   

# ==========================================
# 5. 字段映射 (⚠️ 关键修改)
# ==========================================
# GSM8K 的列名是 'question' 和 'answer'
# MATH  的列名通常是 'problem'  和 'solution'
FIELD_MAP = {
    'question': 'problem',        
    'answer': 'solution' 
}

# ==========================================
# ⚙️ 自动路径生成 (勿动)
# ==========================================
dataset_tag = DATASET_NAME.split('/')[-1]
corpus_file = f"{dataset_tag}_corpus.jsonl"
test_file = f"{dataset_tag}_test_data.jsonl"
index_dir = f"{dataset_tag}_bm25_index"

timestamp = time.strftime("%Y%m%d_%H%M%S")
RESULT_LOG_FILE = f"{dataset_tag}_{MODEL_SOURCE}_{EXPERIMENT_MODE}_{timestamp}.txt"
VIS_IMAGE_FILE = f"memory_distribution_{timestamp}.png"

# 🔥 新增：记忆调用频次排序结果（jsonl）
MEM_FREQ_JSONL_FILE = f"{dataset_tag}_memory_freq_{timestamp}.jsonl"
# ==========================================
# 1. 数据准备模块 (适配 GSM8K)
# ==========================================
def prepare_data():
    print(f"📥 [Step 1] 正在加载数据集: {DATASET_NAME} (Config: {DATASET_CONFIG})...")
    try:
        if DATASET_CONFIG:
            dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
        else:
            dataset = load_dataset(DATASET_NAME)
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

    q_col = FIELD_MAP['question']
    a_col = FIELD_MAP['answer']

    # --- A. 构建记忆库 (使用 Train 集) ---
    if not os.path.exists(corpus_file):
        print(f"🔨 [Memory] 正在将 {SPLIT_TRAIN} 集转换为记忆库: {corpus_file}...")
        with open(corpus_file, "w", encoding="utf-8") as f:
            if SPLIT_TRAIN not in dataset:
                print(f"⚠️ 警告: 数据集没有 {SPLIT_TRAIN} 划分！")
                return False
                
            for i, item in enumerate(tqdm(dataset[SPLIT_TRAIN])):
                q_text = item.get(q_col, "")
                a_text = item.get(a_col, "") # GSM8K 直接是字符串，不需要 eval
                
                # 构建检索内容：通常检索相似的问题和解题思路
                content = f"Question: {q_text}\nAnswer: {a_text}"
                f.write(json.dumps({"id": str(i), "contents": content}) + "\n")
    else:
        print(f"✅ [Memory] 检测到现有记忆库: {corpus_file}，跳过构建。")
    
    # --- B. 准备测试集 ---
    print(f"🔨 [Test] 正在提取测试集 (样本数: {DEBUG_NUM if DEBUG_NUM else 'ALL'})...")
    with open(test_file, "w", encoding="utf-8") as f:
        if SPLIT_TEST not in dataset:
             print(f"❌ 错误: 数据集没有 {SPLIT_TEST} 划分！")
             return False
             
        test_data = dataset[SPLIT_TEST]
        if DEBUG_NUM:
            limit = min(DEBUG_NUM, len(test_data))
            test_data = test_data.select(range(limit))
            
        for i, item in enumerate(test_data):
            q_text = item.get(q_col, "")
            raw_ans = item.get(a_col, "") # 这里是包含 reasoning + #### result 的完整答案
            
            f.write(json.dumps({
                "id": str(i),
                "question": q_text,
                "golden_answers": [str(raw_ans)] # 存入列表保持格式一致
            }) + "\n")
    return True

# ==========================================
# 2. 索引构建模块 (BM25)
# ==========================================
def build_index():
    if os.path.exists(index_dir) and os.path.exists(os.path.join(index_dir, "vocab.tokenizer.json")):
        print(f"✅ [Index] 索引已存在: {index_dir}，跳过构建。")
        return

    print(f"🔨 [Index] 正在为 {corpus_file} 构建 BM25 索引...")
    corpus_texts = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            corpus_texts.append(json.loads(line)['contents'])
    
    corpus_tokens = bm25s.tokenize(corpus_texts)
    retriever_builder = bm25s.BM25()
    retriever_builder.index(corpus_tokens)
    retriever_builder.save(index_dir)
    
    with open(os.path.join(index_dir, "stopwords.tokenizer.json"), "w") as f:
        json.dump([], f)
    with open(os.path.join(index_dir, "vocab.tokenizer.json"), "w") as f:
        vocab = corpus_tokens.vocab
        json.dump({"word_to_id": vocab, "stem_to_sid": vocab, "word_to_stem": {k: k for k in vocab}}, f)
    print("✅ 索引构建完成！")

# ==========================================
# 3. Gemini 生成器类
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
                import time
                time.sleep(2) # 稍微减少一点 sleep，加快速度
            except Exception as e:
                print(f"⚠️ Gemini API Error: {e}")
                import time
                time.sleep(5)
                responses.append("Error")
        return responses

# ==========================================
# 4. 评估工具 (专为 Mathlighteval 改造)
# ==========================================

import re

def extract_math_answer(text):
    """
    专门用于 MATH/LightEval 数据集的答案提取逻辑。
    目标：提取 \boxed{...} 中的内容。
    """
    if not text:
        return None
    text = str(text)

    # --- 策略 1: 标准 \boxed{...} 提取 (支持嵌套括号) ---
    # 简单的正则 r'\\boxed\{(.*?)\}' 无法处理 \boxed{\frac{1}{2}} 这种嵌套情况
    # 所以我们需要从后往前找 \boxed{，然后用栈逻辑匹配右括号
    idx = text.rfind("\\boxed{")
    if idx != -1:
        # 从 "boxed{" 后面开始找
        content_start = idx + 7 
        balance = 0
        for i in range(content_start, len(text)):
            char = text[i]
            if char == '{':
                balance += 1
            elif char == '}':
                if balance == 0:
                    return text[content_start:i] # 找到闭合点，返回内容
                balance -= 1
    
    # --- 策略 2: 如果没找到 boxed，尝试提取最后一行 (保底策略) ---
    # 很多模型如果没有遵循指令输出 boxed，答案通常在最后
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        # 简单的清理：去掉 "Answer:" "The answer is" 等前缀
        last_line = re.sub(r'^(The )?Answer( is)?:?', '', last_line, flags=re.IGNORECASE).strip()
        # 如果剩下的是个很短的字符串（比如数字或短公式），就当做答案
        if len(last_line) < 50: 
            return last_line

    return None

def normalize_latex(s):
    """
    对 LaTeX 答案进行简单的归一化，以便进行字符串比较。
    """
    if not s: return ""
    s = str(s)
    # 1. 去除所有空白字符 (空格、换行) -> "x + y" 变成 "x+y"
    s = "".join(s.split())
    # 2. 统一部分 LaTeX 写法 (可选，根据需要扩展)
    # 比如把 \dfrac 变成 \frac
    s = s.replace("\\dfrac", "\\frac")
    # 3. 去掉文本模式标记 (可选)
    s = s.replace("\\text", "")
    return s.strip()

def evaluate_results(results, experiment_name):
    correct = 0
    total = len(results)
    
    with open(RESULT_LOG_FILE, "a", encoding="utf-8") as f:
        header = f"\n{'='*20} {experiment_name} {'='*20}\n"
        print(header.strip())
        f.write(header)
        
        for i, item in enumerate(results):
            pred = item.pred if hasattr(item, 'pred') else item['pred']
            gold_raw = item.golden_answers[0] if hasattr(item, 'golden_answers') else item['golden_answers'][0]
            question = item.question if hasattr(item, 'question') else item['question']

            # --- 1. 提取 Gold Answer ---
            # MATH 数据集的 gold_raw 通常本身就是 solution，最后一段含有 \boxed{}
            gold_val = extract_math_answer(gold_raw)
            if gold_val is None:
                # 如果标准答案里竟然没有 boxed (极少见)，则取最后一部分
                gold_val = str(gold_raw).strip()

            # --- 2. 提取 Prediction Answer ---
            pred_val = extract_math_answer(pred)
            
            # --- 3. 对比判断 (核心修改：字符串归一化对比) ---
            is_right = False
            
            # 必须两者都有值才能比
            if gold_val and pred_val:
                # 归一化处理
                norm_gold = normalize_latex(gold_val)
                norm_pred = normalize_latex(pred_val)
                
                # 字符串全等对比 (Exact Match)
                if norm_gold == norm_pred:
                    is_right = True
                
                # [可选] 尝试数值对比 (防止 1/2 != 0.5 的情况)
                # 只有当两者看起来都像纯数字时才尝试
                # try:
                #     if abs(float(norm_gold) - float(norm_pred)) < 1e-6:
                #         is_right = True
                # except:
                #     pass

            if is_right:
                correct += 1

            # 打印日志
            log_entry = (
                f"\n[ID]: {i}\n"
                f"[Question]: {str(question)[:100]}...\n" # 题目太长截断一下
                f"[Gold Raw]: ... => [Extracted]: {gold_val}\n"
                f"[Pred Raw]: ...{str(pred)[-50:].replace(chr(10), ' ')} => [Extracted]: {pred_val}\n"
                f"[Result]: {'✅ Correct' if is_right else '❌ Wrong'}\n"
                f"{'-'*30}\n"
            )
            f.write(log_entry)
            if i < 10: print(log_entry.strip())

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
# 🔥 [重构版] 记忆调用频次分析 (含全量统计 & 占位符)
# ==========================================
def analyze_memory_usage(rag_results):
    print("\n🔍 [Analysis] 正在进行全量记忆热度统计...")
    
    # -------------------------------------------------------
    # 1. 建立全量记忆账本 (初始化所有 ID 为 0)，同时保存内容
    # -------------------------------------------------------
    all_memory_ids = set()
    id_to_content = {}  # 新增：记录每条记忆的原始文本

    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                mid = str(item['id'])
                all_memory_ids.add(mid)
                # 记住这条记忆的内容，方便后面写入 jsonl
                id_to_content[mid] = item.get("contents", "")
    except Exception as e:
        print(f"⚠️ 无法读取记忆库文件 {corpus_file}，将仅统计被检索到的记忆。错误: {e}")
    
    # 初始化计数器，所有已知 ID 默认为 0
    memory_counter = collections.Counter({mid: 0 for mid in all_memory_ids})
    
    # -------------------------------------------------------
    # 2. 统计实际检索命中
    # -------------------------------------------------------
    for item in rag_results:
        retrieved_docs = getattr(item, 'retrieval_result', [])
        
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                doc_id = str(doc.get('id'))
            else:
                doc_id = str(getattr(doc, 'id', None))
                
            if doc_id:
                memory_counter[doc_id] += 1

    # -------------------------------------------------------
    # 3. 排序 (按频次降序 -> ID 升序)
    # -------------------------------------------------------
    sorted_memories = sorted(memory_counter.items(), key=lambda x: (-x[1], x[0]))
    
    total_memories = len(sorted_memories)
    used_memories = sum(1 for _, v in sorted_memories if v > 0)
    unused_memories = total_memories - used_memories
    
    print(f"📊 记忆库总量: {total_memories}")
    print(f"🔥 被激活的记忆: {used_memories} ({(used_memories/total_memories)*100:.2f}%)")
    print(f"🧊 沉睡的记忆(0次): {unused_memories}")

    # -------------------------------------------------------
    # 4. ✅ 新增：导出按频次排序的 jsonl
    # -------------------------------------------------------
    try:
        print(f"💾 [Save] 正在导出记忆调用频次排序结果到: {MEM_FREQ_JSONL_FILE}")
        with open(MEM_FREQ_JSONL_FILE, "w", encoding="utf-8") as f:
            for rank, (mid, freq) in enumerate(sorted_memories, start=1):
                record = {
                    "rank": rank,
                    "memory_id": mid,
                    "freq": int(freq),
                    "contents": id_to_content.get(mid, "")
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("✅ 调用频次 jsonl 导出完成！")
    except Exception as e:
        print(f"❌ 导出 {MEM_FREQ_JSONL_FILE} 失败: {e}")

    # -------------------------------------------------------
    # 5. 可视化逻辑 (Top 30 ... Bottom 30)
    # -------------------------------------------------------
    if VISUALIZE_MEMORY_DISTRIBUTION:
        print(f"🎨 [Visual] 正在生成频次分布图: {VIS_IMAGE_FILE}")
        try:
            ids = [m[0] for m in sorted_memories]
            counts = [m[1] for m in sorted_memories]
            
            display_limit = 30
            
            if len(ids) > display_limit * 2:
                print(f"ℹ️ 展示策略: Top {display_limit} + 占位符 + Bottom {display_limit}")
                
                head_ids = ids[:display_limit]
                head_counts = counts[:display_limit]
                
                tail_ids = ids[-display_limit:]
                tail_counts = counts[-display_limit:]
                
                plot_ids = head_ids + ["..."] + tail_ids
                plot_counts = head_counts + [0] + tail_counts
                
                colors = ['skyblue'] * len(head_ids) + ['white'] + ['salmon'] * len(tail_ids)
                edge_colors = ['navy'] * len(head_ids) + ['white'] + ['darkred'] * len(tail_ids)
            else:
                plot_ids = ids
                plot_counts = counts
                colors = 'skyblue'
                edge_colors = 'navy'

            plt.figure(figsize=(15, 6))
            bars = plt.bar(plot_ids, plot_counts, color=colors, edgecolor=edge_colors)
            
            plt.title(f'Memory Usage Distribution (Top {display_limit} vs Bottom {display_limit})', fontsize=14)
            plt.xlabel('Memory ID', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            
            plt.xticks(rotation=90, fontsize=8) 
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if plot_ids[i] != "...": 
                    plt.text(
                        bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=8
                    )
            
            plt.tight_layout()
            plt.savefig(VIS_IMAGE_FILE, dpi=300)
            print("✅ 图片保存成功！")
            
        except ImportError:
            print("❌ 缺少 matplotlib")
            
    else:
        print("\n🏆 [Top 10 Hot Memories]")
        for mid, count in sorted_memories[:10]:
            print(f"   ID: {mid:<5} | Count: {count}")
            
        print("\n🧊 [Bottom 10 Cold Memories]")
        for mid, count in sorted_memories[-10:]:
             print(f"   ID: {mid:<5} | Count: {count}")

# ==========================================
# 5. 主程序
# ==========================================
def main():

    print("Visible GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}:", torch.cuda.get_device_name(i))


    if os.path.exists(RESULT_LOG_FILE): os.remove(RESULT_LOG_FILE)
    print(f"📝 结果将保存至: {RESULT_LOG_FILE}")
    print(f"🛠️ 模式: {EXPERIMENT_MODE} | 源: {MODEL_SOURCE} | 数据集: {DATASET_NAME}")

    if not prepare_data(): return
    if EXPERIMENT_MODE in ['rag', 'all']: build_index()
    
    generator = None
    config = None
    
    if MODEL_SOURCE == "gemini":
        print(f"🤖 [Init] 初始化 Gemini: {GEMINI_MODEL_NAME}...")
        gemini_config_dict = {
            "device": "cpu",
            "retrieval_method": "bm25",
            "corpus_path": corpus_file,
            "index_path": index_dir,
            "retriever_model_path": index_dir,
            "generator_model": "huggingface", 
            "generator_model_path": "gpt2",   
            "generation_method": "custom",  
            "save_dir": "rag_result_cache"
        }
        config = Config(config_dict=gemini_config_dict)
        generator = GeminiGenerator(GEMINI_MODEL_NAME, GEMINI_API_KEY)
        
    elif MODEL_SOURCE == "huggingface":
        print(f"📥 [Init] 检查/下载 HF 模型: {HF_MODEL_NAME}...")
        try:
            model_path = snapshot_download(repo_id=HF_MODEL_NAME)
        except:
            print("❌ 模型下载失败，请检查网络或 HF_ENDPOINT 设置")
            return

        hf_config_dict = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_num": torch.cuda.device_count(),
            "generator_model": "huggingface",
            "generator_model_path": model_path,
            "generation_method": "huggingface",
            "batch_size":10, # 保守一点，防止显存爆炸或 padding 问题
            "max_input_len": 4096, 
            "max_new_tokens": 1024,
            "save_dir": "rag_result_cache"
        }
        print("🚀 加载 HF 生成器...")
        config = Config(config_dict=hf_config_dict)
        generator = get_generator(config)
        
        # 🔥🔥🔥 Critical Fix: Tokenizer & Padding 🔥🔥🔥
        # 很多 Instruct 模型在批量生成时，如果 padding 设置不对，会直接输出乱码或停止符
        if hasattr(generator, 'tokenizer'):
            # 确保使用 Left Padding (生成任务必须)
            generator.tokenizer.padding_side = 'left' 
            # 确保 pad_token 存在
            if generator.tokenizer.pad_token is None:
                generator.tokenizer.pad_token = generator.tokenizer.eos_token
                generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id
            
            # 强制更新 max length
            generator.tokenizer.model_max_length = 4096
        
        if hasattr(generator, 'model'):
            # 确保模型 config 也有 pad token id
            if hasattr(generator.model.config, 'pad_token_id') and generator.model.config.pad_token_id is None:
                generator.model.config.pad_token_id = generator.tokenizer.pad_token_id

        generator.max_input_len = 4096
        print(f"✅ Tokenizer 修正: padding_side='left', pad_token={generator.tokenizer.pad_token}")
    
    else:
        print(f"❌ 未知的 MODEL_SOURCE: {MODEL_SOURCE}")
        return

    with open(test_file, "r") as f:
        test_dataset_raw = [json.loads(line) for line in f]

    acc_baseline = 0
    acc_rag = 0

    # ==========================================
    # 🔥 Prompt 格式化 (修复: 使用纯文本指令格式，放弃容易出错的 ChatML)
    # ==========================================
    def format_base_prompt(system_text, user_text):
        """
        使用最稳健的 Alpaca 风格或 Question/Answer 风格。
        对于 Qwen-Instruct，### Question: ... 这种格式通常比错误的 ChatML 标签更好用。
        """
        if MODEL_SOURCE == "gemini":
            return f"{system_text}\n\n{user_text}" if system_text else user_text
            
        # HuggingFace 模型通用格式
        prompt = ""
        if system_text:
            prompt += f"{system_text}\n\n"
        
        # 构造清晰的 Q&A 结构
        # 重点：末尾加上 "Let's think step by step." 作为 Priming (启动子)
        prompt += f"### Question:\n{user_text}\n\n### Answer:\nLet's think step by step."
        return prompt

    if EXPERIMENT_MODE in ['baseline', 'all']:
        print("\n⚔️ [Task A] 正在运行 Baseline ...")
        
        baseline_inputs = []
        for item in test_dataset_raw:
            # 在 User Text 里不要加 instruction，放到 format 函数里统一加，避免混乱
            user_content = item['question']
            
            # 构造提示词
            sys_msg = "You are a math expert. Solve the problem in a brief. Don't answer more than 50 words.End your answer with \boxed\{number\}."
            formatted_prompt = format_base_prompt(sys_msg, user_content)
            baseline_inputs.append(formatted_prompt)

        baseline_preds = generator.generate(baseline_inputs)
        
        baseline_results = []
        for item, pred in zip(test_dataset_raw, baseline_preds):
            baseline_results.append({
                "question": item['question'],
                "golden_answers": item['golden_answers'],
                "pred": pred
            })
        acc_baseline = evaluate_results(baseline_results, "Baseline (No RAG)")

    if EXPERIMENT_MODE in ['rag', 'all']:
        print("\n⚔️ [Task B] 正在运行 FlashRAG (Few-shot Retrieval)...")
        
        rag_config_dict = config.config_dict.copy() if hasattr(config, 'config_dict') else {}
        if not rag_config_dict:
             rag_config_dict = gemini_config_dict if MODEL_SOURCE == "gemini" else hf_config_dict
             
        rag_config_dict.update({
            "retrieval_method": "bm25",
            "corpus_path": corpus_file,
            "index_path": index_dir,
            "retriever_model_path": index_dir,
            "topk": 3 
        })
        
        rag_config = Config(config_dict=rag_config_dict)
        retriever = get_retriever(rag_config)
        
        # --- FlashRAG Prompt Template (修复版) ---
        # 1. 放弃复杂的 ChatML
        # 2. 统一使用 ### 标记
        # 3. 强制 Priming ("Let's think step by step.")
        
        rag_system_part = (
            "You are a math expert. You can solve math problems in one second to give the correct answer. Below are some similar solved problems. "
            "Refer to the logic in these examples to solve the new question.\n\n"
            "Solve the problem in a very brief. Don't answer more than 80 tokens. If the problem is easy, You can also just give the final answer."
            "Do not perform unit conversion."
            "{reference}\n\n"
            "### Question:\n{question}\n\n"
            "### Answer:\n"
            "Let's think step by step." 
        )
        
        # System Prompt 包含了所有结构，User Prompt 留空（或者由 FlashRAG 内部处理）
        # FlashRAG 的 Dataset 默认把 query 填入 {question}，retrieval 填入 {reference}
        prompt_tpl = PromptTemplate(rag_config, system_prompt=rag_system_part, user_prompt="")
        
        pipeline = SequentialPipeline(rag_config, prompt_template=prompt_tpl, retriever=retriever, generator=generator)
        
        dataset_obj = Dataset(rag_config, test_file)
        
        # 运行 Pipeline
        rag_results = pipeline.run(dataset_obj)
        
        # 1. 评估准确率
        acc_rag = evaluate_results(rag_results, f"FlashRAG ({dataset_tag} Memory)")
        
        # 2. 🔥 统计记忆热度
        analyze_memory_usage(rag_results)

    if EXPERIMENT_MODE == 'all':
        summary = (
            f"\n{'='*20} 最终对比结果 {'='*20}\n"
            f"📊 数据集: {DATASET_NAME}\n"
            f"🤖 模型: {MODEL_SOURCE} / {GEMINI_MODEL_NAME if MODEL_SOURCE=='gemini' else HF_MODEL_NAME}\n"
            f"📉 Baseline: {acc_baseline:.2f}%\n"
            f"📈 FlashRAG: {acc_rag:.2f}%\n"
            f"🚀 提升: {acc_rag - acc_baseline:+.2f}%\n"
            f"{'='*50}\n"
        )
        print(summary)
        with open(RESULT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(summary)

if __name__ == "__main__":
    main()