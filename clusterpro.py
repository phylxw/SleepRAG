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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA # 🔥 重新加回 PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
import umap


# ================= 配置区域 =================

# 1. 核心开关: 选择起名字的模型来源
MODEL_SOURCE = "huggingface" 

# [HuggingFace 配置]
HF_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507" 

# [Gemini 配置]
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 2. 文件配置
INPUT_FILE = "MATH-lighteval_memory_freq_20251218_150403.jsonl" 
OUTPUT_FILE = "AMATH-lighteval_auto_clustered_result.jsonl"
SUMMARY_OUTPUT_FILE = "AMATH-lighteval_cluster_summary.jsonl"
PLOT_FILE = "AMATH-lighteval_cluster_distribution.png"
# 可视化图片输出路径
VIS_PLOT_FILE = "AMATH-lighteval_visualization.png"

# 3. 聚类算法设置 (决定怎么“分”类)
# 选项: 'agglomerative' (自动发现类别数) 或 'kmeans' (指定类别数)
CLUSTERING_METHOD = "agglomerative" 

# [Agglomerative 参数]
DISTANCE_THRESHOLD = 1.0  

# [K-Means 参数]
KMEANS_N_CLUSTERS = 10    

# 4. 可视化降维算法设置 (决定怎么“画”图)
# 选项: 'tsne' (最常用，效果好), 'pca' (最快，线性), 'umap' (平衡，需安装umap-learn)
VISUALIZATION_METHOD = "tsne"

# 5. 数据预处理与高级参数 (🔥 新增：解决聚类“糊成一团”的优化项)
# -------------------------------------------------------------
# 是否在聚类和画图前，先对 Embedding 进行 PCA 降维去噪？
# 推荐: True。通常 Sentence Embedding 维度很高(1024维)，直接聚类效果不好。
# 降维到 50 维左右通常能去除噪音，显著改善 t-SNE 的分离效果。
ENABLE_PCA_PREPROCESS = True
PCA_PREPROCESS_DIMS = 50 

# t-SNE 困惑度 (Perplexity): 
# 控制 t-SNE 关注局部还是全局。数据点多时(>1000)建议调大 (30-50)，少时调小 (5-20)。
# 调整这个参数往往能把"糊成一团"的数据拉开。
TSNE_PERPLEXITY = 40
# -------------------------------------------------------------

# [Embedding 模型]
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5" 
# ===========================================

GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None

# =============== 0. 工具函数 ===============
def clean_special_chars(text: str) -> str:
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

    elif MODEL_SOURCE == "huggingface":
        if GLOBAL_MODEL is None:
            return "Skipped (Model Not Loaded)"
        try:
            print("  🚀 [Local] 正在推理...", end="", flush=True)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = GLOBAL_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = GLOBAL_TOKENIZER([text], return_tensors="pt").to(GLOBAL_MODEL.device)

            with torch.no_grad():
                generated_ids = GLOBAL_MODEL.generate(model_inputs.input_ids, max_new_tokens=50, do_sample=False)
            
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
            
            ids.append(str(obj["memory_id"]))
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

def preprocess_embeddings_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """
    🔥 新增预处理函数: 使用 PCA 降维去噪
    """
    print(f"🧹 正在执行 PCA 预处理 (降维: {embeddings.shape[1]} -> {n_components})...")
    if embeddings.shape[0] < n_components:
        print(f"⚠️ 样本数 ({embeddings.shape[0]}) 少于目标维度 ({n_components})，跳过 PCA 预处理。")
        return embeddings
    
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    
    # 打印保留的方差比例，让用户知道损失了多少信息
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"   >>> 保留方差比例: {explained_variance:.2%}")
    return reduced

def cluster_questions_auto(embeddings: np.ndarray) -> np.ndarray:
    if CLUSTERING_METHOD == "kmeans":
        print(f"🤖 正在执行 K-Means 聚类 (N_Clusters={KMEANS_N_CLUSTERS})...")
        model = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=42, n_init='auto')
        labels = model.fit_predict(embeddings)
        print(f"✨ K-Means 聚类完成！共生成 {KMEANS_N_CLUSTERS} 个类别。")
        return labels
        
    elif CLUSTERING_METHOD == "agglomerative":
        print(f"🤖 正在执行层次聚类 Agglomerative (Distance Threshold={DISTANCE_THRESHOLD})...")
        model = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=DISTANCE_THRESHOLD,
            metric='euclidean', 
            linkage='ward'
        )
        labels = model.fit_predict(embeddings)
        n_clusters_found = len(set(labels))
        print(f"✨ 层次聚类完成！模型自动发现了 {n_clusters_found} 个题型类别。")
        return labels
    
    else:
        raise ValueError(f"未知的聚类方法: {CLUSTERING_METHOD}")

# =============== 4. 统计绘图 & 关键词 & 降维可视化 ===============

def plot_cluster_stats(labels: np.ndarray, save_path: str):
    print(f"\n📊 正在生成统计图表...")
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    singleton_mask = counts == 1
    num_singletons = np.sum(singleton_mask)
    
    valid_mask = ~singleton_mask
    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]
    
    print(f"   - 总聚类数: {len(unique_labels)}")
    print(f"   - 孤立聚类数 (Size=1): {num_singletons}")
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
    plt.title(f'Cluster Size Distribution (Descending)\nMethod: {CLUSTERING_METHOD}', fontsize=14)
    if len(x_ticks) > 30: plt.xticks(rotation=90, fontsize=8)
    else: plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"🖼️ 图表已保存至: {save_path}")

def plot_dimensionality_reduction(embeddings: np.ndarray, labels: np.ndarray, method: str, save_path: str):
    """
    🔥 统一的降维可视化函数，支持 t-SNE, PCA, UMAP
    """
    print(f"\n🎨 正在生成 {method.upper()} 聚类分布图...")
    if embeddings.shape[0] < 2:
        print("⚠️ 数据点太少，跳过可视化。")
        return

    reducer = None
    
    # --- 1. 选择算法 ---
    if method == "tsne":
        n_samples = embeddings.shape[0]
        # 允许用户通过全局参数 TSNE_PERPLEXITY 调整
        perplexity_val = min(TSNE_PERPLEXITY, n_samples - 1) if n_samples > 1 else 1
        print(f"   >>> 运行 t-SNE (perplexity={perplexity_val})...")
        
        reducer = TSNE(
            n_components=2, 
            perplexity=perplexity_val, 
            random_state=42, 
            init='pca', 
            learning_rate='auto'
        )
        
    elif method == "pca":
        print(f"   >>> 运行 PCA (Linear)...")
        reducer = PCA(n_components=2)
        
    elif method == "umap":
        if umap is None:
            print("❌ 未检测到 UMAP 库。请运行 `pip install umap-learn` 安装。")
            print("   (将自动回退到 t-SNE)")
            return plot_dimensionality_reduction(embeddings, labels, "tsne", save_path)
        print(f"   >>> 运行 UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        
    else:
        print(f"❌ 未知的可视化方法: {method}")
        return

    # --- 2. 降维 ---
    reduced_emb = reducer.fit_transform(embeddings)

    # --- 3. 绘图 ---
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        reduced_emb[:, 0], 
        reduced_emb[:, 1], 
        c=labels, 
        cmap='nipy_spectral', 
        s=15, 
        alpha=0.6,
        edgecolor='none'
    )
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'{method.upper()} Visualization\n(Cluster: {CLUSTERING_METHOD}, Preprocess: {ENABLE_PCA_PREPROCESS})', fontsize=15)
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"🖼️ 可视化图表已保存至: {save_path}")

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
    init_llm()

    ids, questions = load_questions(INPUT_FILE)
    if not ids: return

    embeddings = build_embeddings(questions, model_name=EMBEDDING_MODEL)
    
    # 🔥 1. 预处理 (新增步骤：降维去噪)
    if ENABLE_PCA_PREPROCESS:
        embeddings = preprocess_embeddings_pca(embeddings, n_components=PCA_PREPROCESS_DIMS)

    # 2. 聚类
    labels = cluster_questions_auto(embeddings)

    # 3. 画图
    plot_cluster_stats(labels, save_path=PLOT_FILE)
    
    # 4. 可视化 (t-SNE/PCA/UMAP)
    plot_dimensionality_reduction(embeddings, labels, method=VISUALIZATION_METHOD, save_path=VIS_PLOT_FILE)

    # 5. 分析关键词与保存
    keywords_map = tfidf_keywords_per_cluster(questions, labels)
    
    print("\n" + "="*20 + " 聚类结果分析 " + "="*20)
    cluster_labels_text = {}
    
    unique, counts = np.unique(labels, return_counts=True)
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
            
    print(f"💾 保存聚类摘要到 {SUMMARY_OUTPUT_FILE}...")
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
    
    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for cid in sorted(cluster_aggregation.keys()):
            f.write(json.dumps(cluster_aggregation[cid], ensure_ascii=False) + "\n")
            
    print("✅ 全部完成！")

if __name__ == "__main__":
    cluster()