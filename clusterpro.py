import os
import json
import re
import time
import numpy as np
import torch
import matplotlib.pyplot as plt 
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA 
from transformers import AutoModelForCausalLM, AutoTokenizer
import umap

# Hydra
import hydra
from omegaconf import DictConfig

# ================= 全局变量 (保持原逻辑) =================
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_SGLANG_CLIENT = None

# =============== 0. 工具函数 ===============
def clean_special_chars(text: str) -> str:
    if not isinstance(text, str): return text
    return text.replace('\u2028', ' ').replace('\u2029', ' ')

def normalize_text(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r"\d+(\.\d+)?", " <num> ", x) 
    x = re.sub(r"\s+", " ", x).strip()
    return x

def import_torch_and_check_gpu():
    try: return torch.cuda.is_available()
    except: return False

# =============== 1. LLM 初始化与调用 (保持不变，已适配 SGLang) ===============
def init_llm(cfg: DictConfig):
    global GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_SGLANG_CLIENT
    
    model_source = cfg.model.source
    
    if model_source == "gemini":
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            print(f"🤖 [Init] Gemini API ({cfg.model.gemini_name}) 已配置")
        else:
            print("⚠️ [Init] 未检测到 GEMINI_API_KEY，Gemini 模式可能无法工作")
            
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
            print("✅ [Init] 本地模型加载完成！")
        except Exception as e:
            print(f"❌ [Init] 本地模型加载失败: {e}")

    elif model_source == "sglang":
        try:
            from openai import OpenAI
            api_url = cfg.model.get("sglang_api_url", "http://127.0.0.1:30000/v1")
            # 兼容 yaml 里配置 sglang_api_key 的情况，虽然默认是 EMPTY
            api_key = "EMPTY" 
            
            GLOBAL_SGLANG_CLIENT = OpenAI(base_url=api_url, api_key=api_key)
            print(f"✅ [Init] SGLang Client 已连接至 {api_url}")
        except ImportError:
            print("❌ [Init] 缺少 openai 库，请运行 `pip install openai`")

def call_llm(prompt: str, cfg: DictConfig) -> str:
    model_source = cfg.model.source
    
    if model_source == "gemini":
        import google.generativeai as genai
        if not os.environ.get("GEMINI_API_KEY"): return "Skipped (No Key)"
        model = genai.GenerativeModel(cfg.model.gemini_name) 
        try:
            print("  🤖 [Gemini] 正在思考...", end="", flush=True)
            resp = model.generate_content(prompt)
            print(" 完成!")
            return clean_special_chars(resp.text.strip())
        except Exception as e:
            print(f"\n❌ [Gemini Error]: {e}")
            time.sleep(1)
            return "Unknown Topic"

    elif model_source == "huggingface":
        if GLOBAL_MODEL is None: return "Skipped (Model Not Loaded)"
        try:
            print("  🚀 [Local] 正在推理...", end="", flush=True)
            messages = [{"role": "user", "content": prompt}]
            text = GLOBAL_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = GLOBAL_TOKENIZER([text], return_tensors="pt").to(GLOBAL_MODEL.device)
            with torch.no_grad():
                generated_ids = GLOBAL_MODEL.generate(model_inputs.input_ids, max_new_tokens=50, do_sample=False)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = GLOBAL_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(" 完成!")
            return clean_special_chars(response.strip())
        except Exception as e:
            print(f"\n❌ [Local Error]: {e}")
            return "Unknown Topic"

    elif model_source == "sglang":
        if GLOBAL_SGLANG_CLIENT is None: return "Skipped (Client Not Initialized)"
        model_name = cfg.model.get("sglang_model_name", "Qwen/Qwen3-4B-Instruct-2507")
        try:
            print("  🚀 [SGLang] 正在推理...", end="", flush=True)
            resp = GLOBAL_SGLANG_CLIENT.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )
            content = resp.choices[0].message.content
            print(" 完成!")
            return clean_special_chars(content.strip())
        except Exception as e:
            print(f"\n❌ [SGLang Error]: {e}")
            return "Unknown Topic"

    return "Unknown Config"

# =============== 2. 基础 IO (增强健壮性) ===============
def load_questions(jsonl_path: str):
    print(f"📥 正在加载文件: {jsonl_path}...")
    if not os.path.exists(jsonl_path):
        print(f"❌ 找不到文件: {jsonl_path}")
        return [], [], []

    ids, questions, raw_contents = [], [], []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError: continue
            
            # 从 pre.py 的输出中获取内容
            content = obj.get("contents", "")
            raw_contents.append(content) 

            # 🔥 增强的分隔逻辑：兼容 Question/Problem/Input 等不同前缀
            # 如果是 pre.py 生成的，格式固定是 "Question: ... \nAnswer: ..."
            if "Question:" in content:
                # 尝试分离 Question 和 Answer，只聚类 Question 部分
                q_part = content.split("Answer:")[0].replace("Question:", "").strip()
                # 尝试分离 Question 和 Answer，只聚类 Answer 部分
                # q_part = content.split("Question:")[0].replace("Answer:", "").strip()
                if not q_part and content: q_part = content 
            else:
                q_part = content


            mid = obj.get("memory_id", obj.get("id"))
            ids.append(str(mid))
            questions.append(clean_special_chars(q_part))
            
    print(f"✅ 加载完成，共 {len(questions)} 条数据")
    return ids, questions, raw_contents

# =============== 3. 主流程：embedding + 自动聚类 ===============
def build_embeddings(questions: List[str], model_name: str, device_cfg: str = "cuda") -> np.ndarray:
    print(f"🚀 正在计算 Embeddings ({model_name})...")
    
    # 优先使用 config 里的 device，如果不可用则自动检测
    device = device_cfg if (device_cfg == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"   >>> 使用设备: {device}")
    
    model = SentenceTransformer(model_name, device=device)
    q_norm = [normalize_text(q) for q in questions]
    
    # 增大一点 batch_size 提高速度
    emb = model.encode(q_norm, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb)

def preprocess_embeddings_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    print(f"🧹 正在执行 PCA 预处理 (降维: {embeddings.shape[1]} -> {n_components})...")
    if embeddings.shape[0] < n_components:
        print(f"⚠️ 样本数 ({embeddings.shape[0]}) 少于目标维度，跳过 PCA。")
        return embeddings
    
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"   >>> 保留方差比例: {explained_variance:.2%}")
    return reduced

def cluster_questions_auto(embeddings: np.ndarray, cfg: DictConfig) -> np.ndarray:
    method = cfg.cluster.method
    
    if method == "kmeans":
        n_clusters = cfg.cluster.kmeans_n_clusters
        print(f"🤖 正在执行 K-Means 聚类 (N_Clusters={n_clusters})...")
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = model.fit_predict(embeddings)
    elif method == "agglomerative":
        threshold = cfg.cluster.distance_threshold
        print(f"🤖 正在执行层次聚类 (Threshold={threshold})...")
        model = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=threshold,
            metric='euclidean', 
            linkage='ward'
        )
        labels = model.fit_predict(embeddings)
        print(f"✨ 层次聚类发现 {len(set(labels))} 个类别。")
    else:
        raise ValueError(f"未知的聚类方法: {method}")
        
    return labels

# =============== 4. 统计绘图 & 关键词 & 降维可视化 ===============

def plot_cluster_stats(labels: np.ndarray, save_path: str, method_name: str):
    print(f"\n📊 正在生成统计图表...")
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # 只绘制数量 > 1 的聚类
    valid_mask = counts > 1
    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]
    
    if len(valid_counts) == 0:
        print("   ⚠️ 没有包含多个问题的聚类，跳过绘图。")
        return

    # 排序
    sorted_indices = np.argsort(valid_counts)[::-1]
    sorted_plot_labels = valid_labels[sorted_indices]
    sorted_plot_counts = valid_counts[sorted_indices]
    
    plt.figure(figsize=(12, 6))
    x_ticks = [str(lbl) for lbl in sorted_plot_labels]
    # 限制展示数量，防止太密
    if len(x_ticks) > 50:
        x_ticks = x_ticks[:50]
        sorted_plot_counts = sorted_plot_counts[:50]
        
    plt.bar(x_ticks, sorted_plot_counts, color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Cluster ID')
    plt.ylabel('Count')
    plt.title(f'Top Cluster Size Distribution ({method_name})')
    plt.xticks(rotation=90 if len(x_ticks) > 20 else 0, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"🖼️ 图表已保存至: {save_path}")

def plot_dimensionality_reduction(embeddings: np.ndarray, labels: np.ndarray, cfg: DictConfig, save_path: str):
    method = cfg.cluster.vis_method
    print(f"\n🎨 正在生成 {method.upper()} 可视化...")
    
    if embeddings.shape[0] < 5: return

    reducer = None
    if method == "tsne":
        perp = min(cfg.cluster.tsne_perplexity, embeddings.shape[0] - 1)
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    elif method == "pca":
        reducer = PCA(n_components=2)
    elif method == "umap":
        if umap is None:
            print("❌ 缺少 umap-learn，回退到 t-SNE")
            cfg.cluster.vis_method = "tsne"
            return plot_dimensionality_reduction(embeddings, labels, cfg, save_path)
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        print(f"❌ 未知可视化方法: {method}")
        return

    reduced_emb = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    # 使用 jet 或 tab20 这种颜色区分度高的 colormap
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=labels, cmap='tab20', s=10, alpha=0.6)
    plt.colorbar(label='Cluster ID')
    plt.title(f'{method.upper()} Visualization')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"🖼️ 可视化保存至: {save_path}")

def tfidf_keywords_per_cluster(questions, cluster_labels, max_features=5000, top_k=10):
    print("🔍 提取关键词...")
    q_norm = [normalize_text(q) for q in questions]
    try:
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=max_features, stop_words="english")
        X = vectorizer.fit_transform(q_norm)
        vocab = np.array(vectorizer.get_feature_names_out())
        
        cluster_keywords = {}
        for cid in np.unique(cluster_labels):
            idx = np.where(cluster_labels == cid)[0]
            if len(idx) < 2: continue # 太少的就不提取了
            tfidf_mean = np.asarray(X[idx].mean(axis=0)).ravel()
            top_idx = tfidf_mean.argsort()[::-1][:top_k]
            cluster_keywords[cid] = vocab[top_idx].tolist()
        return cluster_keywords
    except ValueError:
        return {}

def llm_label_cluster(cid, questions, cluster_labels, cluster_keywords, cfg: DictConfig, max_examples=5):
    idx = np.where(cluster_labels == cid)[0]
    # 随机采样几个作为示例
    examples_idx = np.random.choice(idx, min(len(idx), max_examples), replace=False)
    examples = [questions[i] for i in examples_idx]
    kw = ", ".join(cluster_keywords.get(cid, []))

    prompt = f"""You are a Math Education Expert.
I have grouped similar math problems together.
Keywords: [{kw}]
Examples:
{chr(10).join(f"- {q}" for q in examples)}

Task: Provide a **very short category name** (3-6 words) for this problem type.
Output ONLY the category name. Do not explain.
"""
    return call_llm(prompt, cfg).replace('"', "").strip()

# =============== Main (Hydra Integrated) ===============

@hydra.main(version_base=None, config_path="conf", config_name="config")
def cluster(cfg: DictConfig):
    
    # 0. 初始化
    init_llm(cfg)

    # 1. 路径映射 (从 yaml 读取)
    # 🔥 关键修改：输入文件现在是 memory_freq (由 pre.py 生成)
    input_file = cfg.paths.freq_file  
    output_file = cfg.paths.cluster_output
    summary_file = cfg.paths.cluster_summary
    plot_file = cfg.paths.cluster_plot
    vis_plot_file = cfg.paths.cluster_vis
    
    # 自动创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)

    ids, questions, raw_contents = load_questions(input_file)
    if not ids: return

    # 2. Embedding
    embeddings = build_embeddings(questions, cfg.model.embedding_name, cfg.model.device)
    
    # 2.1 PCA 预处理
    if cfg.cluster.enable_pca_preprocess:
        embeddings = preprocess_embeddings_pca(embeddings, n_components=cfg.cluster.pca_preprocess_dims)

    # 3. 聚类
    labels = cluster_questions_auto(embeddings, cfg)

    # 4. 统计与可视化
    plot_cluster_stats(labels, save_path=plot_file, method_name=cfg.cluster.method)
    plot_dimensionality_reduction(embeddings, labels, cfg, save_path=vis_plot_file)

    # 5. 分析关键词与命名
    keywords_map = tfidf_keywords_per_cluster(questions, labels)
    
    print("\n" + "="*20 + " 聚类结果分析 " + "="*20)
    unique, counts = np.unique(labels, return_counts=True)
    sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
    
    cluster_labels_text = {}
    print(f"📊 总共发现 {len(sorted_clusters)} 个聚类。")
    print("   (正在为 Top 10 热门聚类生成 LLM 命名...)\n")

    for cid, count in sorted_clusters[:10]: # 只给前10个最大的命名，省 token
        label_text = llm_label_cluster(cid, questions, labels, keywords_map, cfg)
        cluster_labels_text[cid] = label_text
        print(f"   🏷️ Cluster {cid} ({count} 题): {label_text}")
        if cfg.model.source == "gemini": time.sleep(1)

    # 6. 保存详细结果 (每条数据都带 cluster_id)
    print(f"\n💾 保存详细结果到 {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for qid, q, raw, cid in zip(ids, questions, raw_contents, labels):
            obj = {
                "id": qid,
                "contents": raw,  
                "cluster_id": int(cid),
                "cluster_label": cluster_labels_text.get(int(cid), f"Cluster {cid}"),
                "cluster_keywords": keywords_map.get(int(cid), [])
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            
    # 7. 保存摘要结果 (Cluster 为主键)
    print(f"💾 保存聚类摘要到 {summary_file}...")
    cluster_aggregation = {}
    for qid, cid in zip(ids, labels):
        cid_int = int(cid)
        if cid_int not in cluster_aggregation:
            cluster_aggregation[cid_int] = {
                "cluster_id": cid_int,
                "cluster_label": cluster_labels_text.get(cid_int, f"Cluster {cid_int}"),
                "count": 0,
                "memory_ids": []
            }
        cluster_aggregation[cid_int]["memory_ids"].append(qid)
        cluster_aggregation[cid_int]["count"] += 1
    
    with open(summary_file, "w", encoding="utf-8") as f:
        for cid in sorted(cluster_aggregation.keys(), key=lambda k: cluster_aggregation[k]['count'], reverse=True):
            f.write(json.dumps(cluster_aggregation[cid], ensure_ascii=False) + "\n")
            
    print("✅ 全部完成！")

if __name__ == "__main__":
    cluster()