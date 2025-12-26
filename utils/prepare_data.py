from datasets import load_dataset
from omegaconf import DictConfig
import os
import json
from tqdm import tqdm
import random 

def _get_available_column(dataset, candidates, default):
    """辅助函数：在数据集里自动寻找存在的列名"""
    # dataset 可能是 Dataset 对象，也有可能是 dict (如果在 stream 模式)
    # 优先检查 features 或 column_names
    cols = []
    if hasattr(dataset, "column_names"):
        cols = dataset.column_names
    elif hasattr(dataset, "features"):
        cols = dataset.features.keys()
    
    # 遍历候选列表，谁在就用谁
    for cand in candidates:
        if cand in cols:
            return cand
    return default

def prepare_data(cfg: DictConfig, corpus_file: str, test_file: str):
    """
    通用数据准备函数 (支持 GPQA 选择题模式 + 智能列名探测)
    """
    
    # 1. 获取 yaml 里的默认配置 (优先用于 Memory)
    q_col_cfg = cfg.experiment.field_map.question
    a_col_cfg = cfg.experiment.field_map.answer
    
    # 定义探测列表
    q_candidates = [q_col_cfg, "problem", "question", "input", "content", "Question"]
    a_candidates = [a_col_cfg, "solution", "answer", "ground_truth", "output", "completion", "Correct Answer"]

    # ==========================================
    # Part A: 构建记忆库 (Corpus) -> 保持原样
    # ==========================================
    c_name = cfg.experiment.get("corpus_dataset_name") or cfg.experiment.get("dataset_name")
    c_config = cfg.experiment.get("corpus_dataset_config") or cfg.experiment.get("dataset_config")
    c_split = cfg.experiment.get("corpus_split", "train")

    if not os.path.exists(corpus_file):
        print(f"\n🔨 [Memory] 正在构建记忆库: {c_name} | Split: {c_split}")
        try:
            ds_corpus = load_dataset(c_name, c_config, split=c_split)
        except Exception as e:
            print(f"❌ 记忆库下载失败: {e}")
            return False
            
        q_col_mem = _get_available_column(ds_corpus, q_candidates, q_col_cfg)
        a_col_mem = _get_available_column(ds_corpus, a_candidates, a_col_cfg)
        print(f"   👉 自动匹配列名: Q='{q_col_mem}', A='{a_col_mem}'")

        with open(corpus_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(tqdm(ds_corpus, desc="Writing Corpus")):
                q_text = item.get(q_col_mem, "")
                a_text = item.get(a_col_mem, "")
                if q_text:
                    content = f"Question: {q_text}\nAnswer: {a_text}"
                    f.write(json.dumps({"id": str(i), "contents": content}) + "\n")
    else:
        print(f"✅ [Memory] 检测到现有记忆库: {corpus_file}")

    # ==========================================
    # Part B: 准备测试集 (Test) -> 🔥 核心修改在这里
    # ==========================================
    t_name = cfg.experiment.get("test_dataset_name") or c_name
    t_config = cfg.experiment.test_dataset_config if "test_dataset_config" in cfg.experiment else c_config
    t_split = cfg.experiment.get("test_split", "test")

    print(f"\n🔨 [Test] 正在处理测试集: {t_name} | Split: {t_split}")
    try:
        ds_test = load_dataset(t_name, t_config, split=t_split)
    except Exception as e:
        print(f"❌ 测试集下载失败: {e}")
        return False

    # 🔥 判断是否为 GPQA (通过数据集名字判断)
    is_gpqa = "gpqa" in t_name.lower()

    # 自动探测列名 (非 GPQA 时使用)
    if not is_gpqa:
        q_col_test = _get_available_column(ds_test, q_candidates, q_col_cfg)
        a_col_test = _get_available_column(ds_test, a_candidates, a_col_cfg)
        print(f"   👉 自动匹配列名: Q='{q_col_test}', A='{a_col_test}'")
    else:
        print(f"   👉 [Mode] 检测到 GPQA 数据集，切换为选择题处理模式...")

    # --- 切片与写入 ---
    with open(test_file, "w", encoding="utf-8") as f:
        start_idx = int(cfg.model.get("start_index", 0) or 0)
        debug_num = cfg.model.get("debug_num")
        
        total_len = len(ds_test)
        if debug_num:
            limit = int(debug_num)
            end_idx = min(start_idx + limit, total_len)
        else:
            end_idx = total_len
            
        indices = range(start_idx, end_idx)
        selected_data = ds_test.select(indices)
        
        print(f"📊 写入数量: {len(selected_data)}")

        for i, item in enumerate(selected_data):
            real_id = start_idx + i
            
            # 🔥 [关键修改] IF-ELSE 逻辑分支
            if is_gpqa:
                # === 分支 1: GPQA 选择题逻辑 ===
                # 1. 获取原始字段
                question_raw = item.get("Question", "")
                correct_ans = item.get("Correct Answer", "")
                inc_ans_1 = item.get("Incorrect Answer 1", "")
                inc_ans_2 = item.get("Incorrect Answer 2", "")
                inc_ans_3 = item.get("Incorrect Answer 3", "")
                
                # 2. 组合选项并打乱
                options = [correct_ans, inc_ans_1, inc_ans_2, inc_ans_3]
                random.shuffle(options)
                
                # 3. 确定正确选项的字母 (A/B/C/D)
                labels = ['A', 'B', 'C', 'D']
                try:
                    correct_idx = options.index(correct_ans)
                    final_ans = labels[correct_idx] # 答案就是 A, B, C 或 D
                except ValueError:
                    final_ans = "Error"

                # 4. 构造带选项的问题文本
                options_str = ""
                for label, content in zip(labels, options):
                    options_str += f"\n({label}) {content}"
                
                q_text = question_raw + options_str
                a_text = final_ans # 存入 golden_answers 的是字母

            else:
                # === 分支 2: 普通填空题逻辑 (MATH/GSM8K) ===
                q_text = item.get(q_col_test, "")
                a_text = item.get(a_col_test, "")

            # 统一写入
            f.write(json.dumps({
                "id": str(real_id),
                "question": q_text,
                "golden_answers": [str(a_text)]
            }) + "\n")
            
    return True