from datasets import load_dataset
from omegaconf import DictConfig
import os
import json
from tqdm import tqdm
import random 
# 保持原有的工具导入
from tools.prepare.merge_hmmt import merge_hmmt
from tools.prepare.merge_aime import merge_aime
from tools.prepare.sci_split import prepare_sciknow
from tools.prepare.humaneval_split import prepare_humaneval
from tools.prepare.mbpp_split import prepare_mbpp

def _get_available_column(dataset, candidates, default):
    """辅助函数：在数据集里自动寻找存在的列名"""
    cols = []
    if hasattr(dataset, "column_names"):
        cols = dataset.column_names
    elif hasattr(dataset, "features"):
        cols = dataset.features.keys()
    
    for cand in candidates:
        if cand in cols:
            return cand
    return default

def _format_eval_item(item, q_col, a_col, mode="standard"):
    """
    辅助函数：统一处理评估集的数据格式化（GPQA/SciKnow/Standard）
    返回: (question_text, answer_text)
    """
    q_text = ""
    a_text = ""
    
    if mode == "sciknow":
        # SciKnowEval 逻辑
        question_raw = item.get("question", "")
        choices = item.get("choices", []) 
        answer_raw = item.get("answer", "")
        options_str = ""
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
        if isinstance(choices, list):
            for idx, choice_text in enumerate(choices):
                label = labels[idx] if idx < len(labels) else str(idx)
                options_str += f"\n({label}) {choice_text}"
        else:
            options_str = f"\n{str(choices)}"
        q_text = question_raw + options_str
        a_text = str(answer_raw)

    elif mode == "gpqa":
        # GPQA 选择题逻辑
        question_raw = item.get("Question", "")
        correct_ans = item.get("Correct Answer", "")
        # 获取干扰项
        options = [correct_ans]
        for k in ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
            if item.get(k): options.append(item.get(k))
            
        random.shuffle(options)
        labels = ['A', 'B', 'C', 'D']
        try:
            correct_idx = options.index(correct_ans)
            final_ans = labels[correct_idx] 
        except ValueError:
            final_ans = "Error"

        options_str = ""
        for label, content in zip(labels, options):
            options_str += f"\n({label}) {content}"
        
        q_text = question_raw + options_str
        a_text = final_ans 

    else:
        # Standard / Math 逻辑
        q_text = item.get(q_col, "")
        a_text = item.get(a_col, "")
        
    return q_text, a_text

def prepare_data(cfg: DictConfig, corpus_file: str, test_file: str, need_split):
    """
    重构后的数据准备函数：完全解耦 Memory 和 Eval 的数据源
    """
    # === 0. 特殊数据集处理 (保持不变) ===
    tag = cfg.experiment.tag
    if tag == "sci": return prepare_sciknow(corpus_file, test_file, cfg, need_split)
    if tag == "humaneval": return prepare_humaneval(corpus_file, test_file, cfg, need_split)
    if tag == "mbpp": return prepare_mbpp(corpus_file, test_file, cfg, need_split)
    if tag == "hmmtex": merge_hmmt(test_file, cfg, need_split); return True
    if tag == "aimeex": merge_aime(test_file, cfg, need_split); return True

    # 字段探测配置
    q_col_cfg = cfg.experiment.field_map.question
    a_col_cfg = cfg.experiment.field_map.answer
    q_candidates = [q_col_cfg, "problem", "question", "input", "content", "Question"]
    a_candidates = [a_col_cfg, "solution", "answer", "ground_truth", "output", "completion", "Correct Answer"]

    # ==========================================
    # Part A: 构建记忆库 (Corpus) - 严格读取 corpus_* 配置
    # ==========================================
    # 只有当文件不存在，或者需要强制刷新时才处理
    if not os.path.exists(corpus_file) or cfg.parameters.get("force_process_corpus", False):
        c_name = cfg.experiment.get("corpus_dataset_name")
        c_config = cfg.experiment.get("corpus_dataset_config")
        c_split = cfg.experiment.get("corpus_split", "train")

        print(f"\n📚 [Memory] 正在构建记忆库: {c_name} | Config: {c_config} | Split: {c_split}")
        
        try:
            ds_corpus = load_dataset(c_name, c_config, split=c_split)
        except Exception as e:
            print(f"❌ Memory 数据集加载失败: {e}")
            return False

        # --- 1. 题目类型过滤 (保持原有逻辑) ---
        target_type = cfg.experiment.get("problem_type", "all")
        if target_type and target_type.lower() != "all":
            type_candidates = ["problem_type", "subject", "category", "type"]
            type_col = _get_available_column(ds_corpus, type_candidates, None)
            if type_col:
                print(f"🔍 [Filter] 过滤类型: '{target_type}' (列: {type_col})")
                ds_corpus = ds_corpus.filter(
                    lambda x: x[type_col] is not None and target_type.lower() in str(x[type_col]).lower()
                )
            else:
                print(f"⚠️ [Warning] 未找到类型列，跳过过滤。")

        # --- 2. 总量控制 ---
        max_limit = cfg.parameters.get("total_num", None)
        if max_limit is not None and len(ds_corpus) > int(max_limit):
            print(f"✂️  [Memory] 截取前 {max_limit} 条")
            ds_corpus = ds_corpus.select(range(int(max_limit)))

        # --- 3. 写入 Corpus 文件 ---
        q_col = _get_available_column(ds_corpus, q_candidates, q_col_cfg)
        a_col = _get_available_column(ds_corpus, a_candidates, a_col_cfg)
        
        # 如果需要从corpus里切分一部分给验证集（兼容旧逻辑，或防止val配置为空）
        # 但既然你现在参数分开了，这里默认全量写入
        with open(corpus_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(tqdm(ds_corpus, desc="Writing Corpus")):
                q_text = item.get(q_col, "")
                a_text = item.get(a_col, "")
                if q_text:
                    content = f"Question: {q_text}\nAnswer: {a_text}"
                    f.write(json.dumps({"id": str(i), "contents": content}) + "\n")
    else:
        print(f"✅ [Memory] 记忆库已存在: {corpus_file}")


    # ==========================================
    # Part B: 准备评估集 (Eval) - 根据 need_split 决定是用 Val 还是 Test 配置
    # ==========================================
    # 逻辑：
    # 如果 need_split == True (通常代表验证阶段)，读取 val_* 配置
    # 如果 need_split == False (通常代表测试阶段)，读取 test_* 配置
    # 最终都写入 test_file (因为外部工具通常只认这个文件路径)
    
    is_val = need_split
    
    if is_val:
        t_name = cfg.experiment.get("val_dataset_name")
        t_config = cfg.experiment.get("val_dataset_config")
        t_split = cfg.experiment.get("val_split", "test") # 默认从test split拿验证数据
        mode_label = "Validation"
    else:
        t_name = cfg.experiment.get("test_dataset_name")
        t_config = cfg.experiment.get("test_dataset_config")
        t_split = cfg.experiment.get("test_split", "test")
        mode_label = "Test"


    print(f"\n🎯 [{mode_label}] 正在处理评估集: {t_name} | Split: {t_split}")
    
    try:
        ds_eval = load_dataset(t_name, t_config, split=t_split)
    except Exception as e:
        print(f"❌ {mode_label} 数据集加载失败: {e}")
        return False

    # 探测模式 (GPQA / SciKnow / Standard)
    is_gpqa = "gpqa" in str(t_name).lower()
    is_sciknow = "sci" in str(t_name).lower() or cfg.experiment.tag == "sci"
    
    format_mode = "standard"
    if is_sciknow: format_mode = "sciknow"
    elif is_gpqa: format_mode = "gpqa"
    
    if format_mode == "standard":
        q_col_test = _get_available_column(ds_eval, q_candidates, q_col_cfg)
        a_col_test = _get_available_column(ds_eval, a_candidates, a_col_cfg)
        print(f"   👉 列名匹配: Q='{q_col_test}', A='{a_col_test}'")
    else:
        print(f"   👉 模式激活: {format_mode.upper()}")
        q_col_test, a_col_test = None, None # 特殊模式不需要列名

    # --- 截取与写入 ---
    start_idx = int(cfg.parameters.get("start_index", 0) or 0)
    debug_num = cfg.parameters.get("debug_num")
    
    total_len = len(ds_eval)
    if debug_num:
        limit = int(debug_num)
        end_idx = min(start_idx + limit, total_len)
    else:
        end_idx = total_len
    
    # 防止越界
    if start_idx >= total_len:
         print(f"⚠️ [Warning] start_index ({start_idx}) 超出数据范围 ({total_len})")
         selected_data = []
    else:
        selected_data = ds_eval.select(range(start_idx, end_idx))

    print(f"📊 写入 {mode_label} 文件: {len(selected_data)} 条 (Range: {start_idx}-{end_idx})")

    with open(test_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(selected_data):
            real_id = start_idx + i
            
            # 使用统一的格式化函数
            q_text, a_text = _format_eval_item(item, q_col_test, a_col_test, mode=format_mode)

            f.write(json.dumps({
                "id": str(real_id),
                "question": q_text,
                "golden_answers": [str(a_text)]
            }) + "\n")

    return True