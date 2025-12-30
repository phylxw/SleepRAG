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
            
            display_limit = 50
            
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



import os
import json
import re
import time
import numpy as np
import torch
import google.generativeai as genai
import matplotlib.pyplot as plt 
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering 
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置区域 =================

# 1. 核心开关: 选择起名字的模型来源
# 选项: 'huggingface' (本地显卡跑) 或 'gemini' (谷歌API)
MODEL_SOURCE = "huggingface" 

# [HuggingFace 配置]
HF_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507" 

# [Gemini 配置]
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 2. 文件配置
INPUT_FILE = "MATH-lighteval_corpus.jsonl" #其他的直接gsm8k改math就行了
# INPUT_FILE = "gsm8k_corpus.jsonl"
# 输出文件 1: 详细结果 (每行一道题，包含其类别)
OUTPUT_FILE = "MATH-lighteval_auto_clustered_result.jsonl"
# 输出文件 2: 聚类摘要 (每行一个类，包含该类下所有题号) -> 🔥 新增
SUMMARY_OUTPUT_FILE = "MATH-lighteval_cluster_summary.jsonl"
# 输出文件 3: 统计图表
PLOT_FILE = "MATH-lighteval_cluster_distribution.png"

# 3. 聚类参数
DISTANCE_THRESHOLD = 1.0  # 距离阈值
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5" 
# ===========================================

# 全局变量用于存储本地模型，防止重复加载
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None

# =============== 0. 工具函数 ===============
def clean_special_chars(text: str) -> str:
    """清洗异常字符"""
    if not isinstance(text, str): return text
    return text.replace('\u2028', ' ').replace('\u2029', ' ')

def normalize_text(x: str) -> str:
    x = x.lower()
    x = re.sub(r"\d+(\.\d+)?", " <num> ", x) 
    x = re.sub(r"\s+", " ", x).strip()
    return x

def import_torch_and_check_gpu():
    try: return torch.cuda.is_available()
    except: return False

# =============== 1. LLM 初始化与调用 ===============

def init_llm():
    """初始化 LLM (仅针对本地模型)"""
    global GLOBAL_MODEL, GLOBAL_TOKENIZER
    
    if MODEL_SOURCE == "gemini":
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            print(f"🤖 [Init] Gemini API ({GEMINI_MODEL_NAME}) 已配置")
        else:
            print("⚠️ [Init] 未检测到 GEMINI_API_KEY，Gemini 模式可能无法工作")
            
    elif MODEL_SOURCE == "huggingface":
        print(f"📥 [Init] 正在加载本地模型: {HF_MODEL_NAME} ...")
        try:
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
            GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_NAME,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            ).eval()
            print("✅ [Init] 本地模型加载完成！")
        except Exception as e:
            print(f"❌ [Init] 本地模型加载失败: {e}")
            print("💡 提示: 请确保已通过 `huggingface-cli login` 登录或检查网络")

def call_llm(prompt: str) -> str:
    """统一 LLM 调用接口"""
    
    # --- 分支 A: Gemini ---
    if MODEL_SOURCE == "gemini":
        if not GEMINI_API_KEY: return "Skipped (No Key)"
        model = genai.GenerativeModel(GEMINI_MODEL_NAME) 
        try:
            print("  🤖 [Gemini] 正在思考...", end="", flush=True)
            resp = model.generate_content(prompt)
            print(" 完成!")
            return clean_special_chars(resp.text.strip())
        except Exception as e:
            print(f"\n❌ [Gemini Error]: {e}")
            time.sleep(1)
            return "Unknown Topic"

    # --- 分支 B: HuggingFace (本地) ---
    elif MODEL_SOURCE == "huggingface":
        if GLOBAL_MODEL is None:
            return "Skipped (Model Not Loaded)"
        
        try:
            print("  🚀 [Local] 正在推理...", end="", flush=True)
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = GLOBAL_TOKENIZER.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = GLOBAL_TOKENIZER([text], return_tensors="pt").to(GLOBAL_MODEL.device)

            with torch.no_grad():
                generated_ids = GLOBAL_MODEL.generate(
                    model_inputs.input_ids,
                    max_new_tokens=50, 
                    do_sample=False    
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = GLOBAL_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(" 完成!")
            return clean_special_chars(response.strip())
            
        except Exception as e:
            print(f"\n❌ [Local Error]: {e}")
            return "Unknown Topic"
            
    return "Unknown Config"

# =============== 2. 基础 IO ===============
def load_questions(jsonl_path: str):
    print(f"📥 正在加载文件: {jsonl_path}...")
    if not os.path.exists(jsonl_path):
        print(f"❌ 找不到文件: {jsonl_path}")
        return [], []

    ids, questions = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError: continue
            
            content = obj.get("contents", "")
            if "Question:" in content:
                q_part = content.split("Solution:")[0].replace("Question:", "").strip()
            else:
                q_part = content
            
            ids.append(str(obj["id"]))
            questions.append(clean_special_chars(q_part))
            
    print(f"✅ 加载完成，共 {len(questions)} 条数据")
    return ids, questions

# =============== 3. 主流程：embedding + 自动聚类 ===============
def build_embeddings(questions: List[str], model_name: str) -> np.ndarray:
    print(f"🚀 正在计算 Embeddings ({model_name})...")
    device = "cuda" if import_torch_and_check_gpu() else "cpu"
    print(f"   >>> 使用设备: {device}")
    
    model = SentenceTransformer(model_name, device=device)
    q_norm = [normalize_text(q) for q in questions]
    emb = model.encode(q_norm, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb)

def cluster_questions_auto(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    print(f"🤖 正在执行自动聚类 (Distance Threshold={threshold})...")
    
    model = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=threshold,
        metric='euclidean', 
        linkage='ward'
    )
    labels = model.fit_predict(embeddings)
    
    n_clusters_found = len(set(labels))
    print(f"✨ 自动聚类完成！模型自动发现了 {n_clusters_found} 个题型类别。")
    return labels

# =============== 4. 统计绘图 & 关键词 ===============

def plot_cluster_stats(labels: np.ndarray, save_path: str):
    print(f"\n📊 正在生成统计图表...")
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    singleton_mask = counts == 1
    num_singletons = np.sum(singleton_mask)
    
    valid_mask = ~singleton_mask
    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]
    
    print(f"   - 总聚类数: {len(unique_labels)}")
    print(f"   - 孤立聚类数 (Size=1): {num_singletons} (这部分不画在图里)")
    print(f"   - 有效聚类数 (Size>1): {len(valid_labels)}")
    
    if len(valid_counts) == 0:
        print("   ⚠️ 没有包含多个问题的聚类，跳过绘图。")
        return

    sorted_indices = np.argsort(valid_counts)[::-1]
    sorted_plot_labels = valid_labels[sorted_indices]
    sorted_plot_counts = valid_counts[sorted_indices]
    
    plt.figure(figsize=(12, 6))
    x_ticks = [str(lbl) for lbl in sorted_plot_labels]
    plt.bar(x_ticks, sorted_plot_counts, color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Questions', fontsize=12)
    plt.title(f'Cluster Size Distribution (Descending)\n(Excluding {num_singletons} singleton clusters)', fontsize=14)
    if len(x_ticks) > 30: plt.xticks(rotation=90, fontsize=8)
    else: plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"🖼️ 图表已保存至: {save_path}")

def tfidf_keywords_per_cluster(questions, cluster_labels, max_features=5000, top_k=10):
    print("🔍 提取关键词...")
    q_norm = [normalize_text(q) for q in questions]
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=3, max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(q_norm)
    vocab = np.array(vectorizer.get_feature_names_out())

    cluster_keywords = {}
    for cid in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cid)[0]
        if len(idx) == 0: continue
        tfidf_mean = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = tfidf_mean.argsort()[::-1][:top_k]
        cluster_keywords[cid] = vocab[top_idx].tolist()
    return cluster_keywords

def llm_label_cluster(cid, questions, cluster_labels, cluster_keywords, max_examples=5):
    idx = np.where(cluster_labels == cid)[0]
    examples_idx = np.random.choice(idx, min(len(idx), max_examples), replace=False)
    examples = [questions[i] for i in examples_idx]
    kw = ", ".join(cluster_keywords.get(cid, []))

    prompt = f"""You are a Math Education Expert. 
I have automatically grouped similar math problems together.
Keywords: [{kw}]
Examples:
{chr(10).join(f"- {q}" for q in examples)}

Task: Provide a **very short category name** (3-6 words) for this specific math problem type.
Output ONLY the category name.
"""
    label = call_llm(prompt)
    return label.replace("\n", "").replace('"', "").strip()

# =============== Main ===============
def cluster():
    # 0. 初始化
    init_llm()

    # 1. 加载数据
    ids, questions = load_questions(INPUT_FILE)
    if not ids: return

    # 2. Embedding
    embeddings = build_embeddings(questions, model_name=EMBEDDING_MODEL)
    
    # 3. 自动聚类
    labels = cluster_questions_auto(embeddings, threshold=DISTANCE_THRESHOLD)

    # 4. 画图
    plot_cluster_stats(labels, save_path=PLOT_FILE)

    # 5. 分析关键词
    keywords_map = tfidf_keywords_per_cluster(questions, labels)
    
    print("\n" + "="*20 + " 聚类结果分析 " + "="*20)
    cluster_labels_text = {}
    
    unique, counts = np.unique(labels, return_counts=True)
    # 按数量降序排序
    sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
    
    print(f"📊 总共发现 {len(sorted_clusters)} 个聚类。")
    print("   (仅展示并命名包含题目最多的前 10 个聚类)\n")

    for cid, count in sorted_clusters[:10]:
        print(f"\n🏷️ 分析 Cluster {cid} (包含 {count} 题)...")
        label_text = llm_label_cluster(cid, questions, labels, keywords_map)
        cluster_labels_text[cid] = label_text
        print(f"   >>> 题型: {label_text}")
        print(f"   >>> 关键词: {keywords_map.get(cid, [])}")
        if MODEL_SOURCE == "gemini": time.sleep(2)

    # 6. 保存详细结果 (原功能)
    print(f"\n💾 保存详细结果到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for qid, q, cid in zip(ids, questions, labels):
            obj = {
                "id": qid,
                "question": q,
                "cluster_id": int(cid),
                "cluster_label": cluster_labels_text.get(int(cid), f"Cluster {cid}"),
                "cluster_keywords": keywords_map.get(int(cid), [])
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            
    # 7. 🔥 新增：保存聚类摘要索引表
    print(f"💾 保存聚类摘要到 {SUMMARY_OUTPUT_FILE}...")
    
    # 构造聚合数据 {cluster_id: {label, ids}}
    cluster_aggregation = {}
    for qid, cid in zip(ids, labels):
        cid_int = int(cid)
        if cid_int not in cluster_aggregation:
            cluster_aggregation[cid_int] = {
                "cluster_id": cid_int,
                "cluster_label": cluster_labels_text.get(cid_int, f"Cluster {cid_int}"),
                "memory_ids": []
            }
        cluster_aggregation[cid_int]["memory_ids"].append(qid)
    
    # 写入文件
    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 按 cluster_id 排序写入，方便查看
        for cid in sorted(cluster_aggregation.keys()):
            f.write(json.dumps(cluster_aggregation[cid], ensure_ascii=False) + "\n")
            
    print("✅ 全部完成！")



import os
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai


# ================= 配置区域 =================

# 1. LLM 配置：和你聚类文件保持一致
MODEL_SOURCE = "huggingface"   # "huggingface" 或 "gemini"

HF_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# 2. 文件路径（你可以根据数据集改名；这里先以 MATH 为例）
CLUSTERED_FILE = OUTPUT_FILE           # 聚类后的记忆文件
CLUSTER_SUMMARY_FILE = SUMMARY_OUTPUT_FILE           # 每个类有哪些记忆ID
MEM_FREQ_FILE = MEM_FREQ_JSONL_FILE  # 调用频次文件
OUTPUT_OPTIMIZED_FILE = "MATH-lighteval_optimized_memory_k50.jsonl"   # 输出的新记忆库

# 3. 优化逻辑参数
TOP_K_HIGH = 50                # 作为“高频记忆 anchor”的条目数量（按频次排序）
BOTTOM_K_LOW = 50              # 作为“低频记忆扩写对象”的条目数量（按频次从低到高）
LOW_FREQ_THRESHOLD = 2          # 被高频合并时，如果 freq < 这个阈值就直接删掉
TOP_N_SIMILAR_IN_CLUSTER = 5    # 高频 anchor 在类内选 top-n 相似记忆来合并

# 4. 相似度 embedding 模型
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# ==== 新增：LLM 批量与长度控制 ====
LLM_BATCH_SIZE = 4          # 低频扩写时，一批处理多少条
MAX_NEW_TOKENS = 512        # 生成的最大 token 数（输出长度）
MAX_INPUT_TOKENS = 2048     # 输入的最大 token 数，超过会被截断

# ===========================================

GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None


# =============== 工具函数 ===============

def clean_special_chars(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.replace('\u2028', ' ').replace('\u2029', ' ')


def has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# =============== LLM 初始化与调用 ===============

def init_llm():
    """初始化 LLM（和你的聚类脚本保持一致风格）"""
    global GLOBAL_MODEL, GLOBAL_TOKENIZER

    if MODEL_SOURCE == "gemini":
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            print(f"🤖 [Init] Gemini API ({GEMINI_MODEL_NAME}) 已配置")
        else:
            print("⚠️ [Init] 未检测到 GEMINI_API_KEY，Gemini 相关功能会被跳过")
    elif MODEL_SOURCE == "huggingface":
        print(f"📥 [Init] 正在加载本地模型: {HF_MODEL_NAME} ...")
        try:
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
            GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_NAME,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            ).eval()
            print("✅ [Init] 本地模型加载完成！")
        except Exception as e:
            print(f"❌ [Init] 本地模型加载失败: {e}")
            print("💡 提示: 请检查 HuggingFace 权限和网络")


def call_llm(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """统一的大模型调用接口（Gemini / 本地 Qwen），单条调用"""

    # --- Gemini ---
    if MODEL_SOURCE == "gemini":
        if not GEMINI_API_KEY:
            return "Skipped (No GEMINI_API_KEY)"
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            print("  🤖 [Gemini] 正在生成...", end="", flush=True)
            resp = model.generate_content(prompt)
            print(" 完成")
            return clean_special_chars(resp.text.strip())
        except Exception as e:
            print(f"\n❌ [Gemini Error]: {e}")
            return ""

    # --- HuggingFace 本地 ---
    elif MODEL_SOURCE == "huggingface":
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
                max_length=MAX_INPUT_TOKENS,
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

    return ""


# ==== 新增：批量调用接口 ====
def call_llm_batch(prompts: List[str], max_new_tokens: int = MAX_NEW_TOKENS) -> List[str]:
    """
    批量调用 LLM：
    - HuggingFace：真正 batch generate
    - Gemini：内部循环 call_llm（API 不支持 batch）
    """
    if not prompts:
        return []

    # Gemini：简单循环
    if MODEL_SOURCE == "gemini":
        results = []
        for p in prompts:
            results.append(call_llm(p, max_new_tokens=max_new_tokens))
        return results

    # HuggingFace 本地
    if MODEL_SOURCE == "huggingface":
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

            model_inputs = GLOBAL_TOKENIZER(
                text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
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


# ===== 高频 & 低频记忆的 LLM 操作 =====

def summarize_high_freq_memory(anchor_id: str, group_texts: List[str]) -> str:
    """
    高频记忆类内聚合：给定 anchor + 同类若干条相似记忆，把它们合并成一个更“深”的记忆。
    """
    items_formatted = "\n".join(
        f"[{i+1}] {t}" for i, t in enumerate(group_texts)
    )
    prompt = f"""你是数学助教。下面是一组属于同一题型的记忆条目，它们都来自同一个聚类（同类问题）。
请将它们合并成**一条更完整、更抽象的记忆**，要求：

1. 不改变任何结论，也不要引入新的数值或额外事实。
2. 保留所有关键条件、公式与解题结论。
3. 适当总结共同的解题思路，可以合并重复信息。
4. 用English写成一段或两段连续文本，不要分条列出原题号。

待合并的记忆条目如下：
{items_formatted}
"""
    return call_llm(prompt)


# ==== 修改：拆出一个只构造 prompt 的函数，方便批量 ====
def expand_low_freq_memory_prompt(text: str) -> str:
    """
    构造低频记忆扩写的 prompt（不直接调 LLM）
    """
    prompt = f"""你是数学助教。下面是一条数学题目的记忆内容。

请在 **不改变题目条件和答案、不添加任何新数值或事实** 的前提下，对它进行语义扩写：
1. 可以增加对题目考察点的解释和背景说明。
2. 可以加入同义改写、更多自然语言表述，以便未来更容易被检索到。
3. 输出一段或两段English文本，不要丢失原始信息。

原始记忆：
{text}
"""
    return prompt


def expand_low_freq_memory(text: str) -> str:
    """
    单条低频记忆扩写（保持原接口，内部调用单条 LLM）
    """
    prompt = expand_low_freq_memory_prompt(text)
    return call_llm(prompt)


# =============== 数据加载 ===============

def load_clustered_memories(path: str) -> Tuple[Dict[str, dict], List[str]]:
    """
    读取 *_auto_clustered_result.jsonl
    返回：
      - id -> 记录 dict
      - id_list: 保留原始顺序的 id 列表（方便最后写回）
    """
    memories: Dict[str, dict] = {}
    order: List[str] = []
    print(f"📥 正在加载聚类后的记忆文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            mid = str(obj["id"])
            memories[mid] = obj
            order.append(mid)
    print(f"✅ 共加载 {len(memories)} 条记忆")
    return memories, order


def load_cluster_summary(path: str) -> Dict[int, List[str]]:
    """
    读取 *_cluster_summary.jsonl
    返回：cluster_id -> [memory_ids...]
    """
    cluster_to_ids: Dict[int, List[str]] = {}
    print(f"📥 正在加载聚类摘要文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = int(obj["cluster_id"])
            ids = [str(x) for x in obj.get("memory_ids", [])]
            cluster_to_ids[cid] = ids
    print(f"✅ 共加载 {len(cluster_to_ids)} 个聚类")
    return cluster_to_ids


def load_memory_freq(path: str) -> Dict[str, int]:
    """
    读取调用频次文件 MATH-lighteval_memory_freq_*.jsonl
    预期每行包含 memory_id / id, freq 字段。
    """
    freq_map: Dict[str, int] = {}
    print(f"📥 正在加载记忆频次文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            mid = str(obj.get("memory_id", obj.get("id", "")))
            if not mid:
                continue
            freq = int(obj.get("freq", 0))
            freq_map[mid] = freq
    print(f"✅ 频次记录数: {len(freq_map)}")
    return freq_map


# =============== Embedding & 相似度 ===============

def build_embeddings_for_memories(memories: Dict[str, dict]) -> Dict[str, np.ndarray]:
    """
    对所有记忆构建向量，用于类内相似度计算。
    默认为使用记录中的 "question" 字段；如果你想改成 "contents" 就自己换一下。
    """
    device = "cuda" if has_cuda() else "cpu"
    print(f"🚀 正在计算记忆向量 ({EMBEDDING_MODEL}) on {device}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    ids = list(memories.keys())
    texts = []
    for mid in ids:
        rec = memories[mid]
        text = rec.get("question") or rec.get("contents", "")
        texts.append(text)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    id_to_emb = {mid: embeddings[i] for i, mid in enumerate(ids)}
    print(f"✅ 向量构建完成，共 {len(id_to_emb)} 条")
    return id_to_emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# =============== 高频/低频集合选择 ===============

def select_high_low_ids(
    freq_map: Dict[str, int],
    top_k_high: int,
    bottom_k_low: int,
    low_freq_for_low_only: int = 1
):
    """
    从 freq_map 中选：
      - top_k_high 个最高频作为高频 anchor
      - bottom_k_low 个低频候选（但只保留 freq == low_freq_for_low_only 的）
      - 同时记录所有 freq == 0 的 id 方便之后删除
    """
    items = list(freq_map.items())
    # 高频：按 freq 降序
    sorted_desc = sorted(items, key=lambda x: -x[1])
    high_ids = [mid for mid, f in sorted_desc[:top_k_high]]

    # 低频：按 freq 升序
    sorted_asc = sorted(items, key=lambda x: x[1])
    low_ids = []
    zero_ids = []
    for mid, f in sorted_asc:
        if f == 0:
            zero_ids.append(mid)
            continue
        if f == low_freq_for_low_only:
            low_ids.append(mid)
        if len(low_ids) >= bottom_k_low:
            break

    print(f"🔥 高频 anchor 数量: {len(high_ids)}")
    print(f"🧊 0 次调用的记忆数量: {len(zero_ids)}（之后会删除）")
    print(f"🥶 低频扩写候选(freq={low_freq_for_low_only})数量: {len(low_ids)} (最多 bottom_k={bottom_k_low})")
    return set(high_ids), set(low_ids), set(zero_ids)


# =============== 主优化逻辑 ===============

def optimize_memory():
    # 0. 初始化 LLM
    init_llm()

    # 1. 读入基础数据
    memories, id_order = load_clustered_memories(CLUSTERED_FILE)
    cluster_to_ids = load_cluster_summary(CLUSTER_SUMMARY_FILE)
    freq_map = load_memory_freq(MEM_FREQ_FILE)

    # 为所有记忆补齐频次
    for mid in memories.keys():
        freq_map.setdefault(mid, 0)

    # 2. 选出高频、低频、0 频集合
    high_ids, low_ids, zero_ids = select_high_low_ids(
        freq_map,
        TOP_K_HIGH,
        BOTTOM_K_LOW,
        low_freq_for_low_only=LOW_FREQ_THRESHOLD
    )

    # 3. 准备向量，用于类内相似度
    id_to_emb = build_embeddings_for_memories(memories)

    # 4. 高频：类内聚合（merge）
    merged_consumed_ids = set()      # 被当作“邻居”参与 merge 的记忆 id
    to_delete_ids = set()            # 最终要彻底删除的 id（低频被 merge / 频次为0 等）

    print("\n========== 高频记忆聚合阶段 ==========")
    # 按频次从高到低顺序处理 anchor，避免 rank 低的 anchor 抢走高频邻居
    high_ids_sorted = sorted(list(high_ids), key=lambda x: -freq_map.get(x, 0))

    for anchor_id in high_ids_sorted:
        if anchor_id not in memories:
            continue
        if anchor_id in merged_consumed_ids:
            # 说明已经作为别人 group 的成员了，就不再当 anchor
            continue

        rec_anchor = memories[anchor_id]
        cluster_id = rec_anchor.get("cluster_id")
        if cluster_id is None:
            continue

        cluster_id = int(cluster_id)
        cluster_member_ids = [str(x) for x in cluster_to_ids.get(cluster_id, [])]
        if not cluster_member_ids:
            continue

        # 候选邻居：同类、不是自己、没被 merge 过
        candidates = [
            mid for mid in cluster_member_ids
            if mid != anchor_id and mid not in merged_consumed_ids
        ]
        if not candidates:
            continue

        anchor_emb = id_to_emb.get(anchor_id)
        if anchor_emb is None:
            continue

        sims = []
        for mid in candidates:
            emb = id_to_emb.get(mid)
            if emb is None:
                continue
            sims.append((mid, cosine_similarity(anchor_emb, emb)))

        if not sims:
            continue

        # 取类内 top-n 相似
        sims_sorted = sorted(sims, key=lambda x: -x[1])
        neighbors = [mid for mid, _ in sims_sorted[:TOP_N_SIMILAR_IN_CLUSTER]]
        group_ids = [anchor_id] + neighbors

        print(f"\n🔥 Anchor {anchor_id} (freq={freq_map[anchor_id]}, cluster={cluster_id})")
        print(f"   合并同类 top-{len(neighbors)}: {neighbors}")

        # 构造要给 LLM 的文本
        group_texts = []
        for mid in group_ids:
            rec = memories[mid]
            text = rec.get("question") or rec.get("contents", "")
            group_texts.append(f"[ID {mid}] {text}")

        summary_text = summarize_high_freq_memory(anchor_id, group_texts)
        if not summary_text:
            print("   ⚠️ LLM 返回为空，跳过这组合并")
            continue

        # 更新 anchor 的内容：用 summary 替换 question，并保留原始信息
        original_text = rec_anchor.get("question") or rec_anchor.get("contents", "")
        rec_anchor["original_question"] = original_text
        rec_anchor["question"] = summary_text
        rec_anchor["merged_from_ids"] = group_ids
        rec_anchor["merge_type"] = "high_freq_anchor"

        # 邻居标记为已参与 merge；其中低频的标记为删除
        for mid in neighbors:
            merged_consumed_ids.add(mid)
            if freq_map.get(mid, 0) < LOW_FREQ_THRESHOLD:
                to_delete_ids.add(mid)

    # 5. 低频：不被合并消掉、freq=1 的记忆做扩写
    print("\n========== 低频记忆扩写阶段 ==========")
    # 先把所有 freq=0 的直接加入删除集合
    to_delete_ids.update(zero_ids)

    # 真正要扩写的低频记忆：freq==LOW_FREQ_THRESHOLD，且没有被 merge 消耗掉
    low_expand_ids = [
        mid for mid in low_ids
        if mid in memories and mid not in to_delete_ids
    ]
    print(f"🥶 需要扩写的低频记忆条目数: {len(low_expand_ids)}")

    # ==== 修改：这一段改成批量调用 LLM，而不是一条一条 ====
    low_expand_items = []
    for mid in low_expand_ids:
        rec = memories[mid]
        base_text = rec.get("question") or rec.get("contents", "")
        low_expand_items.append((mid, base_text))

    total_low = len(low_expand_items)
    for start in range(0, total_low, LLM_BATCH_SIZE):
        end = min(start + LLM_BATCH_SIZE, total_low)
        batch_items = low_expand_items[start:end]
        batch_ids = [mid for (mid, _) in batch_items]

        print(f"\n🥶 扩写低频记忆 Batch {start // LLM_BATCH_SIZE + 1} / { (total_low + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE }")
        print(f"   IDs: {batch_ids}")

        batch_prompts = [expand_low_freq_memory_prompt(base_text) for (_, base_text) in batch_items]
        batch_outputs = call_llm_batch(batch_prompts, max_new_tokens=MAX_NEW_TOKENS)

        for (mid, base_text), expanded in zip(batch_items, batch_outputs):
            if not expanded:
                print(f"   ⚠️ LLM 返回为空，ID={mid} 保持原文不变")
                continue
            rec = memories[mid]
            rec["original_question"] = base_text
            rec["question"] = expanded
            rec["opt_type"] = "low_freq_expanded"

    # 6. 写出新的记忆库：跳过 to_delete_ids
    print("\n========== 写出优化后的记忆库 ==========")
    kept_count = 0
    with open(OUTPUT_OPTIMIZED_FILE, "w", encoding="utf-8") as f:
        for mid in id_order:
            if mid not in memories:
                continue
            if mid in to_delete_ids:
                continue
            f.write(json.dumps(memories[mid], ensure_ascii=False) + "\n")
            kept_count += 1

    print(f"✅ 新记忆库写入完成: {OUTPUT_OPTIMIZED_FILE}")
    print(f"   保留记忆条目: {kept_count}")
    print(f"   删除记忆条目: {len(to_delete_ids)}")
    print("   （注意：原始 *_auto_clustered_result.jsonl 文件没有被修改）")



if __name__ == "__main__":
    main()
    cluster()
    optimize_memory()
