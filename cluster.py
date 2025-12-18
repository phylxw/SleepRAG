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

if __name__ == "__main__":
    cluster()