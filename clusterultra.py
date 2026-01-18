import os
import json
import re
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns  # <--- 新增这行
from sentence_transformers import SentenceTransformer

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional deps (safe import)
try:
    import umap  # umap-learn
except Exception:
    umap = None

try:
    import hdbscan  # pip install hdbscan
except Exception:
    hdbscan = None

# Hydra
import hydra
from omegaconf import DictConfig

# ================= 全局变量 (保持原逻辑) =================
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_SGLANG_CLIENT = None


# =============== 0. 工具函数 ===============
def clean_special_chars(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.replace('\u2028', ' ').replace('\u2029', ' ')

def normalize_text(x: str) -> str:
    """
    NOTE: 保持原有行为（小写、数字归一、空格规整），避免改变整体功能。
    如果你数据里有大量中文，可以考虑在这里额外做全角/半角、标点清洗（可选）。
    """
    x = str(x).lower()
    x = re.sub(r"\d+(\.\d+)?", " <num> ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def import_torch_and_check_gpu():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

def _cfg_get(cfg: DictConfig, key: str, default: Any) -> Any:
    """
    OmegaConf 安全取值：兼容旧 config 没有新字段的情况（不会破坏原功能）。
    """
    try:
        return cfg.get(key, default)  # DictConfig supports .get
    except Exception:
        return default


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
            api_key = "EMPTY"
            GLOBAL_SGLANG_CLIENT = OpenAI(base_url=api_url, api_key=api_key)
            print(f"✅ [Init] SGLang Client 已连接至 {api_url}")
        except ImportError:
            print("❌ [Init] 缺少 openai 库，请运行 `pip install openai`")

def call_llm(prompt: str, cfg: DictConfig) -> str:
    model_source = cfg.model.source

    if model_source == "gemini":
        import google.generativeai as genai
        if not os.environ.get("GEMINI_API_KEY"):
            return "Skipped (No Key)"
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
        if GLOBAL_MODEL is None:
            return "Skipped (Model Not Loaded)"
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
        if GLOBAL_SGLANG_CLIENT is None:
            return "Skipped (Client Not Initialized)"
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
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = obj.get("contents", "")
            raw_contents.append(content)

            # 保持原逻辑：默认聚类 Question 部分
            if "Question:" in content:
                q_part = content.split("Answer:")[0].replace("Question:", "").strip()
                if not q_part and content:
                    q_part = content
            else:
                q_part = content

            mid = obj.get("memory_id", obj.get("id"))
            ids.append(str(mid))
            questions.append(clean_special_chars(q_part))

    print(f"✅ 加载完成，共 {len(questions)} 条数据")
    return ids, questions, raw_contents


# =============== 3. 主流程：embedding + 聚类 ===============
def build_embeddings(questions: List[str], model_name: str, device_cfg: str = "cuda") -> np.ndarray:
    print(f"🚀 正在计算 Embeddings ({model_name})...")

    device = device_cfg if (device_cfg == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"   >>> 使用设备: {device}")

    model = SentenceTransformer(model_name, device=device)
    q_norm = [normalize_text(q) for q in questions]

    # normalize_embeddings=True 让向量都落在单位球面上：更适合 cosine / spherical clustering
    emb = model.encode(
        q_norm,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.asarray(emb)

def preprocess_embeddings_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    print(f"🧹 正在执行 PCA 预处理 (降维: {embeddings.shape[1]} -> {n_components})...")
    if embeddings.shape[0] < n_components:
        print(f"⚠️ 样本数 ({embeddings.shape[0]}) 少于目标维度，跳过 PCA。")
        return embeddings

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)

    explained_variance = float(np.sum(pca.explained_variance_ratio_))
    print(f"   >>> 保留方差比例: {explained_variance:.2%}")
    return reduced


def _auto_kmeans(embeddings: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """
    KMeans 自动选 K：用 Silhouette Score 选最优 K（对旧配置兼容）。
    """
    n = embeddings.shape[0]
    k_min = int(_cfg_get(cfg.cluster, "kmeans_k_min", 2))
    k_max = int(_cfg_get(cfg.cluster, "kmeans_k_max", min(50, max(3, int(np.sqrt(n)) + 5))))
    k_max = min(k_max, n - 1)

    if k_max < k_min:
        print("⚠️ 数据太少，回退到 KMeans(n_clusters=2)")
        model = KMeans(n_clusters=2, random_state=42, n_init='auto')
        return model.fit_predict(embeddings)

    sample_size = int(_cfg_get(cfg.cluster, "silhouette_sample", 2000))
    sample_size = min(sample_size, n)

    best_k, best_score, best_labels = None, -1.0, None
    print(f"🔎 [KMeans-Auto] 搜索 K in [{k_min}, {k_max}] (silhouette sample={sample_size}) ...")

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(embeddings)

        # silhouette 要求至少 2 个簇且每簇至少 1 个点
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(
                embeddings, labels,
                metric="euclidean",
                sample_size=sample_size,
                random_state=42
            )
        except Exception:
            continue

        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    if best_labels is None:
        print("⚠️ [KMeans-Auto] 未找到有效 K，回退到 KMeans(n_clusters=cfg.cluster.kmeans_n_clusters)")
        n_clusters = int(cfg.cluster.kmeans_n_clusters)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        return model.fit_predict(embeddings)

    print(f"✅ [KMeans-Auto] 选择 K={best_k}, silhouette={best_score:.4f}")
    return best_labels


def cluster_questions_auto(embeddings: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """
    改进点（不破坏原功能）：
    - 增加 method=hdbscan / method=kmeans_auto
    - 改进 agglomerative：默认用 cosine + average（更契合单位向量 embedding）
    """
    method = cfg.cluster.method

    if method == "kmeans":
        n_clusters = int(cfg.cluster.kmeans_n_clusters)
        print(f"🤖 正在执行 K-Means 聚类 (N_Clusters={n_clusters})...")
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = model.fit_predict(embeddings)

    elif method == "kmeans_auto":
        labels = _auto_kmeans(embeddings, cfg)

    elif method == "agglomerative":
        threshold = float(cfg.cluster.distance_threshold)
        linkage = _cfg_get(cfg.cluster, "agglom_linkage", "average")  # ward / average / complete / single
        metric = _cfg_get(cfg.cluster, "agglom_metric", "cosine")     # euclidean / cosine

        # ward 只支持 euclidean
        if linkage == "ward":
            metric = "euclidean"

        print(f"🤖 正在执行层次聚类 (linkage={linkage}, metric={metric}, threshold={threshold})...")
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric=metric,
            linkage=linkage
        )
        labels = model.fit_predict(embeddings)
        print(f"✨ 层次聚类发现 {len(set(labels))} 个类别。")

    elif method == "hdbscan":
        if hdbscan is None:
            print("❌ 未安装 hdbscan，回退到 agglomerative。请运行: pip install hdbscan")
            cfg.cluster.method = "agglomerative"
            return cluster_questions_auto(embeddings, cfg)

        # HDBSCAN 参数：更适合“簇数未知 + 含噪声”的记忆/文本数据
        min_cluster_size = int(_cfg_get(cfg.cluster, "hdbscan_min_cluster_size", 8))
        min_samples = _cfg_get(cfg.cluster, "hdbscan_min_samples", None)
        metric = _cfg_get(cfg.cluster, "hdbscan_metric", "euclidean")  # euclidean / cosine
        cluster_selection_method = _cfg_get(cfg.cluster, "hdbscan_selection_method", "eom")

        print(f"🤖 正在执行 HDBSCAN 聚类 "
              f"(min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric={metric})...")
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method
        )
        labels = model.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int(np.sum(labels == -1))
        print(f"✨ HDBSCAN 发现 {n_clusters} 个簇；噪声点 {noise}/{len(labels)} (label=-1)。")

    else:
        raise ValueError(f"未知的聚类方法: {method}")

    return labels


# =============== 4. 统计绘图 & 关键词 & 降维可视化 ===============
def plot_cluster_stats(labels: np.ndarray, save_path: str, method_name: str):
    print(f"\n📊 正在生成统计图表...")
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 只绘制数量 > 1 的聚类；HDBSCAN 的噪声(-1)也不绘制
    valid_mask = (counts > 1) & (unique_labels != -1)
    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]

    if len(valid_counts) == 0:
        print("   ⚠️ 没有包含多个问题的聚类，跳过绘图。")
        return

    sorted_indices = np.argsort(valid_counts)[::-1]
    sorted_plot_labels = valid_labels[sorted_indices]
    sorted_plot_counts = valid_counts[sorted_indices]

    plt.figure(figsize=(12, 6))
    x_ticks = [str(lbl) for lbl in sorted_plot_labels]

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
    """
    改进点：
    - 先用 PCA 降到 vis_pca_dims（默认 50）提升 t-SNE/UMAP 稳定性与速度
    - UMAP 默认用 cosine + 更小 min_dist（更容易“同类聚在一起”）
    - 可选 supervised UMAP：用已有 labels 作为 y，让图更“簇状”（只影响可视化，不改变聚类结果）
    """
    method = cfg.cluster.vis_method
    print(f"\n🎨 正在生成 {method.upper()} 可视化...")

    n = embeddings.shape[0]
    if n < 5:
        return

    X = embeddings
    vis_pca_dims = int(_cfg_get(cfg.cluster, "vis_pca_dims", 50))
    if X.shape[1] > vis_pca_dims and vis_pca_dims > 2:
        X = PCA(n_components=vis_pca_dims, random_state=42).fit_transform(X)

    reducer = None
    if method == "tsne":
        perp = min(int(cfg.cluster.tsne_perplexity), n - 1)
        metric = _cfg_get(cfg.cluster, "tsne_metric", "cosine")
        reducer = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=42,
            init='pca',
            learning_rate='auto',
            metric=metric
        )
        reduced_emb = reducer.fit_transform(X)

    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        reduced_emb = reducer.fit_transform(X)

    elif method == "umap":
        if umap is None:
            print("❌ 缺少 umap-learn，回退到 t-SNE。请运行: pip install umap-learn")
            cfg.cluster.vis_method = "tsne"
            return plot_dimensionality_reduction(embeddings, labels, cfg, save_path)

        n_neighbors = int(_cfg_get(cfg.cluster, "umap_n_neighbors", 15))
        min_dist = float(_cfg_get(cfg.cluster, "umap_min_dist", 0.05))
        metric = _cfg_get(cfg.cluster, "umap_metric", "cosine")
        supervised = bool(_cfg_get(cfg.cluster, "umap_supervised", True))
        target_weight = float(_cfg_get(cfg.cluster, "umap_target_weight", 0.5))

        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(n_neighbors, n - 1),
            min_dist=min_dist,
            metric=metric,
            target_metric="categorical",
            target_weight=target_weight if supervised else 0.0
        )

        # supervised: UMAP(X, y=labels) 仅用于图像更聚类化；不会改变 labels 本身
        if supervised and labels is not None:
            reduced_emb = reducer.fit_transform(X, y=labels)
        else:
            reduced_emb = reducer.fit_transform(X)

    else:
        print(f"❌ 未知可视化方法: {method}")
        return

    plt.figure(figsize=(12, 10))

    # 处理 HDBSCAN 噪声点：灰色
    if np.any(labels == -1):
        noise_mask = (labels == -1)
        non_noise = ~noise_mask
        plt.scatter(reduced_emb[noise_mask, 0], reduced_emb[noise_mask, 1], c="lightgray", s=8, alpha=0.5, linewidths=0)
        sc = plt.scatter(reduced_emb[non_noise, 0], reduced_emb[non_noise, 1], c=labels[non_noise], cmap='tab20', s=10, alpha=0.75, linewidths=0)
        plt.colorbar(sc, label='Cluster ID (noise=-1 excluded)')
    else:
        sc = plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=labels, cmap='tab20', s=10, alpha=0.75, linewidths=0)
        plt.colorbar(sc, label='Cluster ID')

    plt.title(f'{method.upper()} Visualization')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"🖼️ 可视化保存至: {save_path}")
    
    # [新增] 返回降维后的数据，供 KDE 使用
    return reduced_emb


def plot_top_clusters_kde(reduced_emb: np.ndarray, labels: np.ndarray, save_path: str, top_k: int = 3):
    """
    挑选数量最多的 top_k 个聚类，在降维后的 2D 平面上绘制 KDE 等高线图。
    (修正版：修复图例不显示的问题)
    """
    print(f"\n🌊 正在生成 Top-{top_k} 聚类的 KDE 密度图...")
    
    if reduced_emb is None or len(reduced_emb) == 0:
        print("⚠️ 降维数据为空，跳过 KDE 绘图。")
        return

    # 1. 统计 Top K 聚类 (排除噪声 -1)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # 过滤掉 -1 和样本数过少的类
    valid_mask = (unique_labels != -1) & (counts >= 5)
    if not np.any(valid_mask):
        print("⚠️ 有效聚类不足，跳过 KDE。")
        return

    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]
    
    # 按数量降序排列
    sorted_indices = np.argsort(valid_counts)[::-1]
    top_labels = valid_labels[sorted_indices][:top_k]
    
    print(f"   目标聚类 ID: {top_labels}")

    # 2. 绘图
    plt.figure(figsize=(10, 8))
    
    # 画背景灰点 (Label = Other)
    # 注意：这里加 alpha=0.3 让背景淡一点，突出前景
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c='lightgray', s=5, alpha=0.3, label='Other')

    # 循环画 Top K 的 KDE 和 散点
    # 使用 seaborn 默认调色盘，或者 tab10
    colors = sns.color_palette("tab10", len(top_labels)) 
    
    for i, cid in enumerate(top_labels):
        # 提取该聚类的点
        mask = (labels == cid)
        subset = reduced_emb[mask]
        
        # 准备图例文本
        label_text = f'Cluster {cid} (n={len(subset)})'
        
        try:
            # 1. 画 KDE (晕染背景) - 也就是那层雾
            # 注意：把 label 去掉，避免图例混乱或不显示
            sns.kdeplot(
                x=subset[:, 0], 
                y=subset[:, 1], 
                fill=True, 
                alpha=0.2,    # 透明度低一点，不要遮住点
                color=colors[i], 
                warn_singular=False
            )
            
            # 2. 画 散点 (实心点) - 把 Label 加在这里！
            # 这样图例里就会出现一个颜色对应的实心圆点，非常清晰
            plt.scatter(
                subset[:, 0], 
                subset[:, 1], 
                s=10, 
                color=colors[i], 
                alpha=0.8, 
                label=label_text  # <--- 关键修改：Label 移到这里
            )
            
        except Exception as e:
            print(f"   ⚠️ Cluster {cid} 画图失败: {e}")

    plt.title(f'KDE Density Plot for Top {len(top_labels)} Clusters')
    
    # 强制显示图例，位置放在最佳位置
    plt.legend(loc='best')
    plt.tight_layout()
    
    # 保存
    kde_save_path = save_path.replace(".png", "_kde.png")
    plt.savefig(kde_save_path, dpi=300)
    print(f"🖼️ KDE 图表已保存至: {kde_save_path}")

def tfidf_keywords_per_cluster(questions, cluster_labels, max_features=5000, top_k=10):
    print("🔍 提取关键词...")
    q_norm = [normalize_text(q) for q in questions]
    try:
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=max_features, stop_words="english")
        X = vectorizer.fit_transform(q_norm)
        vocab = np.array(vectorizer.get_feature_names_out())

        cluster_keywords = {}
        for cid in np.unique(cluster_labels):
            if cid == -1:
                continue
            idx = np.where(cluster_labels == cid)[0]
            if len(idx) < 2:
                continue
            tfidf_mean = np.asarray(X[idx].mean(axis=0)).ravel()
            top_idx = tfidf_mean.argsort()[::-1][:top_k]
            cluster_keywords[cid] = vocab[top_idx].tolist()
        return cluster_keywords
    except ValueError:
        return {}

def llm_label_cluster(cid, questions, cluster_labels, cluster_keywords, cfg: DictConfig, max_examples=5):
    idx = np.where(cluster_labels == cid)[0]
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
    callback = call_llm(prompt, cfg)
    print('我的输出是这样的')
    print(callback)
    return callback.replace('"', "").strip()


# =============== Main (Hydra Integrated) ===============
@hydra.main(version_base=None, config_path="conf", config_name="config")
def cluster(cfg: DictConfig):

    # 0. 初始化
    init_llm(cfg)

    # 1. 路径映射 (从 yaml 读取)
    input_file = cfg.paths.freq_file
    output_file = cfg.paths.cluster_output
    summary_file = cfg.paths.cluster_summary
    plot_file = cfg.paths.cluster_plot
    vis_plot_file = cfg.paths.cluster_vis

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)

    ids, questions, raw_contents = load_questions(input_file)
    if not ids:
        return

    # 2. Embedding
    embeddings = build_embeddings(questions, cfg.model.embedding_name, cfg.model.device)

    # 2.1 PCA 预处理（保持原开关）
    if cfg.cluster.enable_pca_preprocess:
        embeddings = preprocess_embeddings_pca(embeddings, n_components=int(cfg.cluster.pca_preprocess_dims))

    # 3. 聚类（新增 hdbscan / kmeans_auto，但不影响原方法）
    labels = cluster_questions_auto(embeddings, cfg)

    # 4. 统计与可视化
    plot_cluster_stats(labels, save_path=plot_file, method_name=cfg.cluster.method)
    
    # [修改] 接收返回的 reduced_emb
    reduced_emb = plot_dimensionality_reduction(embeddings, labels, cfg, save_path=vis_plot_file)

    # [新增] 调用 KDE 绘图
    # 这里的 top_k 可以写死为 3，或者从 cfg 读取
    if reduced_emb is not None:
        plot_top_clusters_kde(reduced_emb, labels, save_path=vis_plot_file, top_k=3)

    # 5. 分析关键词与命名
    keywords_map = tfidf_keywords_per_cluster(questions, labels)

    print("\n" + "=" * 20 + " 聚类结果分析 " + "=" * 20)
    unique, counts = np.unique(labels, return_counts=True)
    sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)

    cluster_labels_text = {}
    print(f"📊 总共发现 {len(sorted_clusters)} 个聚类（含噪声=-1）。")
    print("   (正在为 Top 10 热门聚类生成 LLM 命名...)\n")

    for cid, count in sorted_clusters[:10]:
        # 对噪声点不命名
        if cid == -1:
            continue
        label_text = llm_label_cluster(cid, questions, labels, keywords_map, cfg)
        cluster_labels_text[int(cid)] = label_text
        print(f"   🏷️ Cluster {cid} ({count} 题): {label_text}")
        if cfg.model.source == "gemini":
            time.sleep(1)

    # 6. 保存详细结果
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

    # 7. 保存摘要结果
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
