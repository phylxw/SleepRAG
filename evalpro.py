import os
import json
import re
import time
import torch
import bm25s
import logging
import ast
import collections
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import snapshot_download
from flashrag.config import Config
from flashrag.pipeline import SequentialPipeline
from flashrag.utils import get_retriever, get_generator, Dataset
from flashrag.prompt import PromptTemplate
import matplotlib.pyplot as plt
import transformers

# 屏蔽 transformers 的冗余警告
transformers.logging.set_verbosity_error()

# ==========================================
# 🛠️ 核心配置区域
# ==========================================

# 1. 实验控制开关
# 这里的 "单纯测试代码" 默认只运行 RAG 模式
EXPERIMENT_MODE = "rag" 

# 2. 记忆库文件配置 (核心修改)
# 指定外部优化过的记忆库文件
MEMORY_SOURCE_FILE = "AMATH-lighteval_optimized_memory_k50.jsonl"

# 3. 结果可视化开关
VISUALIZE_MEMORY_DISTRIBUTION = True

# 4. 调试样本数
# None 表示跑全量测试集，设置数字(如 10)可快速调试
DEBUG_NUM = None

# 5. 模型设置
MODEL_SOURCE = "huggingface" 
HF_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507" 
# HF_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # 备选推荐

# [Gemini 配置]
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 

# ==========================================
# 6. 数据集配置 (测试集来源)
# ==========================================

DATASET_NAME = "DigitalLearningGmbH/MATH-lighteval" 
# MATH 数据集需要指定子集，"all" 表示加载代数、几何等所有类别
DATASET_CONFIG = "algebra" 
SPLIT_TEST = "test"  
FIELD_MAP = {
    'question': 'problem',        
    'answer': 'solution' 
}
# ==========================================
# ⚙️ 自动路径生成
# ==========================================
dataset_tag = DATASET_NAME.split('/')[-1]
# 为了适配 FlashRAG，我们将外部记忆库转换为标准的 corpus.jsonl 格式
corpus_file = f"{dataset_tag}_custom_memory_corpus.jsonl"
test_file = f"{dataset_tag}_test_data.jsonl"
index_dir = f"{dataset_tag}_custom_memory_bm25_index"

timestamp = time.strftime("%Y%m%d_%H%M%S")
RESULT_LOG_FILE = f"{dataset_tag}_{MODEL_SOURCE}_{EXPERIMENT_MODE}_{timestamp}.txt"
VIS_IMAGE_FILE = f"memory_distribution_{timestamp}.png"
MEM_FREQ_JSONL_FILE = f"{dataset_tag}_memory_freq_{timestamp}.jsonl"

# ==========================================
# 1. 数据准备模块 (已修改：读取指定记忆文件)
# ==========================================
def prepare_data():
    print(f"📥 [Step 1] 正在加载测试数据集: {DATASET_NAME} (Config: {DATASET_CONFIG})...")
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

    # --- A. 适配记忆库 (读取 MATH_optimized_memory_k30.jsonl) ---
    # 如果不存在转换后的 corpus 文件，则进行转换
    with open(MEMORY_SOURCE_FILE, "r", encoding="utf-8") as fin, open(corpus_file, "w", encoding="utf-8") as fout:       
        count = 0
        for line in tqdm(fin, desc="Converting Memory"):
            try:
                item = json.loads(line)
                # 这里的 item 结构: {"id": "2", "question": "...\nAnswer:...", "cluster_id": ...}
                # FlashRAG 需要 "contents" 字段用于检索
                # 直接使用 question 字段（因为它包含了问题和答案）
                new_item = {
                    "id": str(item.get("id")),
                    "contents": item.get("question", ""),
                    # 保留其他元数据以备不时之需（可选）
                    "cluster_id": item.get("cluster_id"),
                    "cluster_label": item.get("cluster_label") 
                }
                fout.write(json.dumps(new_item) + "\n")
                count += 1
            except json.JSONDecodeError:
                continue
        print(f"✅ 记忆库转换完成，共处理 {count} 条记忆。")

    # --- B. 准备测试集 (保持不变) ---
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
            raw_ans = item.get(a_col, "") 
            
            f.write(json.dumps({
                "id": str(i),
                "question": q_text,
                "golden_answers": [str(raw_ans)] 
            }) + "\n")
    return True

# ==========================================
# 2. 索引构建模块 (BM25)
# ==========================================
def build_index():
    # 检查索引是否已经存在且匹配
    if os.path.exists(index_dir) and os.path.exists(os.path.join(index_dir, "vocab.tokenizer.json")):
        print(f"✅ [Index] 索引目录已存在: {index_dir}，跳过构建。")
        # 注意：如果更换了记忆文件，建议手动删除 index 文件夹以强制重建
        return

    print(f"🔨 [Index] 正在为 {corpus_file} 构建 BM25 索引...")
    corpus_texts = []
    
    # 读取转换后的标准 corpus 文件
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
                time.sleep(1) 
            except Exception as e:
                print(f"⚠️ Gemini API Error: {e}")
                time.sleep(2)
                responses.append("Error")
        return responses

# ==========================================
# 4. 评估工具
# ==========================================
def extract_math_answer(text):
    if not text:
        return None
    text = str(text)

    # 策略 1: 标准 \boxed{...} 提取
    idx = text.rfind("\\boxed{")
    if idx != -1:
        content_start = idx + 7 
        balance = 0
        for i in range(content_start, len(text)):
            char = text[i]
            if char == '{':
                balance += 1
            elif char == '}':
                if balance == 0:
                    return text[content_start:i] 
                balance -= 1
    
    # 策略 2: 提取最后一行
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        last_line = re.sub(r'^(The )?Answer( is)?:?', '', last_line, flags=re.IGNORECASE).strip()
        if len(last_line) < 50: 
            return last_line

    return None

def normalize_latex(s):
    if not s: return ""
    s = str(s)
    s = "".join(s.split())
    s = s.replace("\\dfrac", "\\frac")
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

            gold_val = extract_math_answer(gold_raw) or str(gold_raw).strip()
            pred_val = extract_math_answer(pred)
            
            is_right = False
            if gold_val and pred_val:
                norm_gold = normalize_latex(gold_val)
                norm_pred = normalize_latex(pred_val)
                if norm_gold == norm_pred:
                    is_right = True

            if is_right:
                correct += 1

            log_entry = (
                f"\n[ID]: {i}\n"
                f"[Question]: {str(question)[:100]}...\n"
                f"[Gold]: {gold_val}\n"
                f"[Pred]: {pred_val}\n"
                f"[Result]: {'✅ Correct' if is_right else '❌ Wrong'}\n"
                f"{'-'*30}\n"
            )
            f.write(log_entry)
            if i < 5: print(log_entry.strip())

        acc = correct / total * 100 if total > 0 else 0
        summary = (
            f"\n📊 统计 ({experiment_name}):\n"
            f"Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}%\n"
            f"{'='*50}\n"
        )
        print(summary)
        f.write(summary)
    return acc


# def extract_last_number(text):
#     """
#     专门用于 GSM8K 的答案提取逻辑。
#     1. 优先寻找 '####' 标记，取其后内容。
#     2. 如果没有标记，使用正则提取文本中的最后一个数字。
#     """
#     text = str(text)
    
#     # 策略 1: 标准 GSM8K 格式分割 (####)
#     if "####" in text:
#         text = text.split("####")[-1]
    
#     # 策略 2: 正则提取数字 (支持整数、浮点数、移除逗号)
#     # 匹配模式: 负号可选，数字，可能有逗号，可能有小数点
#     # 例如: -1,234.56
#     text = text.replace(',', '') # 去掉千分位逗号
#     matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    
#     if matches:
#         return float(matches[-1]) # 返回最后一个数字
#     return None

# def evaluate_results(results, experiment_name):
#     correct = 0
#     total = len(results)
    
#     with open(RESULT_LOG_FILE, "a", encoding="utf-8") as f:
#         header = f"\n{'='*20} {experiment_name} {'='*20}\n"
#         print(header.strip())
#         f.write(header)
        
#         for i, item in enumerate(results):
#             pred = item.pred if hasattr(item, 'pred') else item['pred']
#             gold_raw = item.golden_answers[0] if hasattr(item, 'golden_answers') else item['golden_answers'][0]
#             question = item.question if hasattr(item, 'question') else item['question']

#             # --- 1. 提取 Gold Answer (标准数值) ---
#             # GSM8K 数据集中的 gold_raw 包含 "推理过程 #### 答案"
#             gold_val = extract_last_number(gold_raw)

#             # --- 2. 提取 Prediction Answer (预测数值) ---
#             pred_val = extract_last_number(pred)
            
#             # --- 3. 对比判断 ---
#             is_right = False
#             if gold_val is not None and pred_val is not None:
#                 # 浮点数对比，容差 1e-6
#                 if abs(gold_val - pred_val) < 1e-6:
#                     is_right = True
            
#             if is_right:
#                 correct += 1

#             log_entry = (
#                 f"\n[ID]: {i}\n"
#                 f"[Question]: {question}\n"
#                 f"[Gold Raw]: ...{str(gold_raw)[-50:]} => [Extracted]: {gold_val}\n"
#                 f"[Pred Raw]: ...{str(pred)[-50:].replace(chr(10), ' ')} => [Extracted]: {pred_val}\n"
#                 f"[Result]: {'✅ Correct' if is_right else '❌ Wrong'}\n"
#                 f"{'-'*30}\n"
#             )
#             f.write(log_entry)
#             if i < 10: print(log_entry.strip()) # 只打印前几个防止刷屏

#         acc = correct / total * 100
#         summary = (
#             f"\n📊 统计 ({experiment_name}):\n"
#             f"Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}%\n"
#             f"{'='*50}\n"
#         )
#         print(summary)
#         f.write(summary)
#     return acc

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
# 5. 记忆热度统计
# ==========================================
def analyze_memory_usage(rag_results):
    print("\n🔍 [Analysis] 正在进行全量记忆热度统计...")
    
    all_memory_ids = set()
    id_to_content = {} 

    # 读取转换后的 corpus 文件以建立基准
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                mid = str(item['id'])
                all_memory_ids.add(mid)
                id_to_content[mid] = item.get("contents", "")
    except Exception as e:
        print(f"⚠️ 无法读取记忆库文件 {corpus_file}，统计可能不完整。错误: {e}")
    
    memory_counter = collections.Counter({mid: 0 for mid in all_memory_ids})
    
    for item in rag_results:
        retrieved_docs = getattr(item, 'retrieval_result', [])
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                doc_id = str(doc.get('id'))
            else:
                doc_id = str(getattr(doc, 'id', None))
                
            if doc_id:
                memory_counter[doc_id] += 1

    sorted_memories = sorted(memory_counter.items(), key=lambda x: (-x[1], x[0]))
    
    total_memories = len(sorted_memories)
    used_memories = sum(1 for _, v in sorted_memories if v > 0)
    
    print(f"📊 记忆库总量: {total_memories}")
    print(f"🔥 被激活的记忆: {used_memories} ({(used_memories/total_memories)*100:.2f}%)")

    # 导出 Jsonl
    try:
        with open(MEM_FREQ_JSONL_FILE, "w", encoding="utf-8") as f:
            for rank, (mid, freq) in enumerate(sorted_memories, start=1):
                record = {
                    "rank": rank,
                    "memory_id": mid,
                    "freq": int(freq),
                    "contents": id_to_content.get(mid, "")
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"💾 频次统计已导出: {MEM_FREQ_JSONL_FILE}")
    except Exception as e:
        print(f"❌ 导出失败: {e}")

    # 画图
    if VISUALIZE_MEMORY_DISTRIBUTION:
        try:
            ids = [m[0] for m in sorted_memories]
            counts = [m[1] for m in sorted_memories]
            
            display_limit = 30
            if len(ids) > display_limit * 2:
                plot_ids = ids[:display_limit] + ["..."] + ids[-display_limit:]
                plot_counts = counts[:display_limit] + [0] + counts[-display_limit:]
                colors = ['skyblue'] * display_limit + ['white'] + ['salmon'] * display_limit
                edge_colors = ['navy'] * display_limit + ['white'] + ['darkred'] * display_limit
            else:
                plot_ids = ids
                plot_counts = counts
                colors = 'skyblue'
                edge_colors = 'navy'

            plt.figure(figsize=(15, 6))
            plt.bar(plot_ids, plot_counts, color=colors, edgecolor=edge_colors)
            plt.title(f'Memory Usage (Source: {MEMORY_SOURCE_FILE})', fontsize=14)
            plt.xticks(rotation=90, fontsize=8) 
            plt.tight_layout()
            plt.savefig(VIS_IMAGE_FILE, dpi=300)
            print(f"✅ 分布图已保存: {VIS_IMAGE_FILE}")
        except:
            pass

# ==========================================
# 6. 主程序
# ==========================================
def main():
    if os.path.exists(RESULT_LOG_FILE): os.remove(RESULT_LOG_FILE)
    print(f"📝 结果日志: {RESULT_LOG_FILE}")
    print(f"🛠️ 模式: {EXPERIMENT_MODE} | 记忆源: {MEMORY_SOURCE_FILE}")

    # 准备数据 (包含记忆库转换)
    if not prepare_data(): return
    
    # 总是构建/检查索引，因为 RAG 需要
    build_index()
    
    generator = None
    config = None
    
    if MODEL_SOURCE == "gemini":
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
        try:
            model_path = snapshot_download(repo_id=HF_MODEL_NAME)
        except:
            print("❌ 模型下载失败")
            return

        hf_config_dict = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_num": torch.cuda.device_count(),
            "generator_model": "huggingface",
            "generator_model_path": model_path,
            "generation_method": "huggingface",
            "batch_size": 10,
            "max_input_len": 4096, 
            "max_new_tokens": 1024,
            "save_dir": "rag_result_cache"
        }
        config = Config(config_dict=hf_config_dict)
        generator = get_generator(config)
        
        if hasattr(generator, 'tokenizer'):
            generator.tokenizer.padding_side = 'left' 
            if generator.tokenizer.pad_token is None:
                generator.tokenizer.pad_token = generator.tokenizer.eos_token
                generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id
            generator.tokenizer.model_max_length = 4096
            
        if hasattr(generator, 'model') and hasattr(generator.model.config, 'pad_token_id') and generator.model.config.pad_token_id is None:
            generator.model.config.pad_token_id = generator.tokenizer.pad_token_id

        generator.max_input_len = 4096

    def format_base_prompt(system_text, user_text):
        if MODEL_SOURCE == "gemini":
            return f"{system_text}\n\n{user_text}" if system_text else user_text
        prompt = ""
        if system_text: prompt += f"{system_text}\n\n"
        prompt += f"### Question:\n{user_text}\n\n### Answer:\nLet's think step by step."
        return prompt

    with open(test_file, "r") as f:
        test_dataset_raw = [json.loads(line) for line in f]

    # --- Baseline 任务 (保留但默认不跑，除非 EXPERIMENT_MODE 改为 all) ---
    acc_baseline = 0
    if EXPERIMENT_MODE in ['baseline', 'all']:
        print("\n⚔️ [Task A] Baseline (No RAG) ...")
        baseline_inputs = []
        for item in test_dataset_raw:
            # sys_msg = "You are a math expert. Solve the problem in a brief. Don't answer more than 50 words.End your answer with \\boxed{number}."
            sys_msg = "You are a math expert. Solve the problem in a brief. Don't answer more than 50 words.End your answer with #### <number>."
            
            baseline_inputs.append(format_base_prompt(sys_msg, item['question']))
        
        baseline_preds = generator.generate(baseline_inputs)
        baseline_results = []
        for item, pred in zip(test_dataset_raw, baseline_preds):
            baseline_results.append({
                "question": item['question'],
                "golden_answers": item['golden_answers'],
                "pred": pred
            })
        acc_baseline = evaluate_results(baseline_results, "Baseline")

    # --- RAG 任务 (主要任务) ---
    acc_rag = 0
    if EXPERIMENT_MODE in ['rag', 'all']:
        print("\n⚔️ [Task B] FlashRAG (Memory: Optimized K30) ...")
        
        rag_config_dict = config.config_dict.copy() if hasattr(config, 'config_dict') else {}
        if not rag_config_dict:
             rag_config_dict = gemini_config_dict if MODEL_SOURCE == "gemini" else hf_config_dict
             
        rag_config_dict.update({
            "retrieval_method": "bm25",
            "corpus_path": corpus_file, # 指向转换后的 corpus
            "index_path": index_dir,
            "retriever_model_path": index_dir,
            "topk": 3 
        })
        
        rag_config = Config(config_dict=rag_config_dict)
        retriever = get_retriever(rag_config)
        
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
        
        prompt_tpl = PromptTemplate(rag_config, system_prompt=rag_system_part, user_prompt="")
        pipeline = SequentialPipeline(rag_config, prompt_template=prompt_tpl, retriever=retriever, generator=generator)
        dataset_obj = Dataset(rag_config, test_file)
        
        rag_results = pipeline.run(dataset_obj)
        acc_rag = evaluate_results(rag_results, "FlashRAG w/ Optimized Memory")
        analyze_memory_usage(rag_results)

    print("\n✅ 测试结束。")

if __name__ == "__main__":
    main()