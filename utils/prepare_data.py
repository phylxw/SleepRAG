from datasets import load_dataset
from omegaconf import DictConfig
import os
import json
from tqdm import tqdm
import random 
from tools.prepare.merge_hmmt import merge_hmmt
from tools.prepare.merge_aime import merge_aime
from tools.prepare.sci_split import prepare_sciknow
from tools.prepare.humaneval_split import prepare_humaneval
from tools.prepare.mbpp_split import prepare_mbpp

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

def prepare_data(cfg: DictConfig, corpus_file: str, test_file: str, need_split):
    """
    通用数据准备函数 (支持 GPQA 选择题模式 + 智能列名探测)
    """
    # 先检查sci
    is_val = False #是验证集吗？
    if cfg.experiment.tag == "sci":
        # 直接调用分离出去的模块
        return prepare_sciknow(corpus_file, test_file, cfg, need_split)
    if cfg.experiment.tag == "humaneval":
        # 直接调用分离出去的模块
        return prepare_humaneval(corpus_file, test_file, cfg, need_split)
    if cfg.experiment.tag == "mbpp":
        # 直接调用分离出去的模块
        return prepare_mbpp(corpus_file, test_file, cfg, need_split)
    if (cfg.experiment.tag != "math_self") and (cfg.experiment.tag != "gsm8k_self"):
        is_val = need_split
        need_split = False
        
    
    # 1. 获取 yaml 里的默认配置 (优先用于 Memory)
    q_col_cfg = cfg.experiment.field_map.question
    a_col_cfg = cfg.experiment.field_map.answer
    
    # 定义探测列表
    q_candidates = [q_col_cfg, "problem", "question", "input", "content", "Question"]
    a_candidates = [a_col_cfg, "solution", "answer", "ground_truth", "output", "completion", "Correct Answer"]

    # ==========================================
    # Part A: 构建记忆库 (Corpus) + [新增] 自动切分验证集逻辑
    # ==========================================
    c_name = cfg.experiment.get("corpus_dataset_name") or cfg.experiment.get("dataset_name")
    c_config = cfg.experiment.get("corpus_dataset_config") or cfg.experiment.get("dataset_config")
    c_split = cfg.experiment.get("corpus_split", "train")

    # 🔥 [新增] 读取切分配置
    # 建议在 yaml 里配置: split_ratio (例如 0.9 表示90%做记忆, 10%做验证) 或者 val_num (例如 200)
    split_ratio = cfg.parameters.get("split_ratio", 0.9)
    # val_num = cfg.experiment.get("val_subset_num", 0)  # 想要分出来多少条做验证
    
    # 只要文件不存在 或者 需要强制重新切分(防止用了旧的全量记忆)，就进入处理逻辑
    # 注意：如果启用了 split，建议每次都重新生成，因为涉及到随机切分

    if not os.path.exists(corpus_file) or need_split: 
        print(f"\n🔨 [Memory] 正在处理数据: {c_name} | Split: {c_split}")
        try:
            ds_corpus = load_dataset(c_name, c_config, split=c_split)
        except Exception as e:
            print(f"❌ 数据集下载失败: {e}")
            return False

        # -------------------------------------------------------
        # 🔥 [核心修改] HMMT 专属逻辑: 强制过滤 Level 5
        # -------------------------------------------------------
        target_level = cfg.experiment.get("level_filter", None) # 默认从 yaml 读
        
        if cfg.experiment.tag == "hmmtex" or cfg.experiment.tag == "aimeex":
            print(f"🚀 [Mode] HMMT 模式已激活: 强制过滤 MATH Level 5 数据作为记忆")
            target_level = "Level 5" # 强制指定
            
            # 确保我们取的是 solution (推理过程)
            # 如果 yaml 里没配对，这里强制修正查找列表的优先级
            if "solution" in ds_corpus.column_names:
                a_candidates.insert(0, "solution") 

        # 执行难度过滤
        if target_level:
            level_candidates = ["level", "difficulty", "grade"]
            level_col = _get_available_column(ds_corpus, level_candidates, None)

            if level_col:
                original_len = len(ds_corpus)
                # 过滤逻辑: 只要包含 '5' 或者是 'Level 5'
                ds_corpus = ds_corpus.filter(
                    lambda x: x[level_col] is not None and ("5" in str(x[level_col]))
                )
                print(f"🔥 [Filter] 难度提纯 ({target_level}): {original_len} -> {len(ds_corpus)} 条")
            else:
                print(f"⚠️ [Warning] 未找到难度列，无法执行 Level 5 过滤！")

        # ==================== [新增代码 START] ====================
        # 根据 yaml 中的 problem_type (例如 "Algebra") 进行过滤
        target_type = cfg.experiment.get("problem_type", "all")
        
        if target_type and target_type.lower() != "all":
            print(f"🔍 [Filter] 正在根据题目类型过滤: '{target_type}'")
            
            # 探测题目类型的列名 (OpenR1/MATH 数据集通常是 'subject' 或 'problem_type')
            type_candidates = ["problem_type", "subject", "category", "type"]
            type_col = _get_available_column(ds_corpus, type_candidates, None)
            
            if type_col:
                original_len = len(ds_corpus)
                # 过滤逻辑：检查目标类型是否包含在列值中 (忽略大小写)
                ds_corpus = ds_corpus.filter(
                    lambda x: x[type_col] is not None and target_type.lower() in str(x[type_col]).lower()
                )
                print(f"   👉 过滤结果: {original_len} -> {len(ds_corpus)} 条 (列名: {type_col})")
            else:
                print(f"⚠️ [Warning] 未找到表示题目类型的列，跳过过滤。现有列: {ds_corpus.column_names}")
        # ==================== [新增代码 END] ====================

        # --- 1. 总量控制 (响应你刚才提到的只取前2000条的需求) ---
        max_limit = cfg.parameters.get("total_num", None) # 在 yaml parameters 里加这个参数
        if max_limit is not None and len(ds_corpus) > int(max_limit):
            print(f"✂️  截取前 {max_limit} 条数据进行实验")
            ds_corpus = ds_corpus.select(range(int(max_limit)))

        q_col_mem = _get_available_column(ds_corpus, q_candidates, q_col_cfg)
        a_col_mem = _get_available_column(ds_corpus, a_candidates, a_col_cfg)
        print(f"   👉 自动匹配列名: Q='{q_col_mem}', A='{a_col_mem}'")

        # --- 2. 执行切分逻辑 (核心修改) ---
        if need_split and split_ratio > 0:
            print(f"🔀 [Split] 检测到切分模式: 从 Corpus 中{len(ds_corpus)}条记忆划分 {1 - split_ratio} 的比例作为验证集(Test File)")
            # 打乱数据 (设置固定 seed 保证复现)
            ds_corpus = ds_corpus.shuffle(seed=42)
            
            # 确保数量不越界
            split_idx = int(len(ds_corpus)*split_ratio)
            if split_idx < 0: split_idx = 0
            
            # 切分
            ds_memory = ds_corpus.select(range(0, split_idx)) # 大部分做记忆
            ds_val = ds_corpus.select(range(split_idx, len(ds_corpus))) # 小部分做验证
        else:
            print(f"📦 [Full] 全量模式: 所有数据均用于构建记忆库")
            ds_memory = ds_corpus
            ds_val = None

        # --- 3. 写入记忆库文件 (Corpus File) ---
        if not os.path.exists(corpus_file):
            with open(corpus_file, "w", encoding="utf-8") as f:
                for i, item in enumerate(tqdm(ds_memory, desc="Writing Corpus")):
                    q_text = item.get(q_col_mem, "")
                    a_text = item.get(a_col_mem, "")
                    if q_text:
                        # 记忆库格式: Question/Answer 纯文本
                        content = f"Question: {q_text}\nAnswer: {a_text}"
                        f.write(json.dumps({"id": str(i), "contents": content}) + "\n")
        
        # --- 4. [新增 & 修正] 如果切分了，把验证集写入 Test File (支持 start_index 和 debug_num) ---
        if need_split and ds_val is not None:
            print(f"📝 [Split] 正在将划分出的验证集写入: {test_file}")
            
            # === 👇 新增：读取调试参数 ===
            start_idx = int(cfg.parameters.get("start_index", 0) or 0)
            debug_num = cfg.parameters.get("debug_num")
            
            total_val_len = len(ds_val)
            
            # 计算切片范围
            if debug_num:
                limit = int(debug_num)
                end_idx = min(start_idx + limit, total_val_len)
            else:
                end_idx = total_val_len
            
            # 防止 start_index 越界
            if start_idx >= total_val_len:
                print(f"⚠️ [Warning] start_index ({start_idx}) 超过了验证集总数 ({total_val_len})，将写入空文件。")
                selected_val = []
            else:
                # 对验证集进行切片
                indices = range(start_idx, end_idx)
                selected_val = ds_val.select(indices)
                print(f"📊 [Debug] 验证集截取生效: 范围[{start_idx}:{end_idx}] | 实际写入: {len(selected_val)} 条")

            with open(test_file, "w", encoding="utf-8") as f:
                for i, item in enumerate(tqdm(selected_val, desc="Writing Validation Set")):
                    # === 👇 修正：ID 需要加上偏移量，保持唯一性 ===
                    real_id = start_idx + i 
                    
                    q_text = item.get(q_col_mem, "")
                    a_text = item.get(a_col_mem, "")
                    
                    # 写入符合 Eval 格式的数据
                    f.write(json.dumps({
                        "id": str(real_id),
                        "question": q_text,
                        "golden_answers": [str(a_text)] 
                    }) + "\n")
            
            print(f"✅ [Done] 验证集准备完毕 (已截取)，跳过原始 Test Set 下载步骤")
            return True  # 截断后续逻辑

    else:
        print(f"✅ [Memory] 检测到现有记忆库: {corpus_file}")

    # ==========================================
    # Part B: 准备测试集 (Test)
    # ==========================================
    if cfg.experiment.tag == "hmmtex":
        print(f"✅ 执行多HMMT组合测试文件下载")
        merge_hmmt(test_file, cfg, is_val)
        return True
    
    if cfg.experiment.tag == "aimeex":
        print(f"✅ 执行多AIME组合测试文件下载")
        merge_aime(test_file, cfg, is_val)
        return True
    
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
    is_gpqa = "gpqa" in str(t_name).lower()
    # 注意：其实前面的 tag=="sci" 已经return了，这里 is_sciknow 基本不会触发
    # 但为了逻辑完整性保留也可以，或者删掉
    is_sciknow = "sci" in str(t_name).lower()

    if not is_gpqa and not is_sciknow:
        q_col_test = _get_available_column(ds_test, q_candidates, q_col_cfg)
        a_col_test = _get_available_column(ds_test, a_candidates, a_col_cfg)
        print(f"   👉 自动匹配列名: Q='{q_col_test}', A='{a_col_test}'")
    elif is_sciknow:
        print(f"   👉 [Mode] SciKnowEval 科学模式已激活 (处理 choices 列表)")
    else:
        print(f"   👉 [Mode] GPQA 选择题模式已激活")

    # --- 切片与写入 ---
    with open(test_file, "w", encoding="utf-8") as f:
        start_idx = int(cfg.parameters.get("start_index", 0) or 0)
        debug_num = cfg.parameters.get("debug_num")
        
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

            # 🔥🔥🔥 [关键修复] 使用 if-elif-else 互斥结构
            if is_sciknow:
                # === 分支 1: SciKnowEval 逻辑 ===
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

            elif is_gpqa:
                # === 分支 2: GPQA 选择题逻辑 ===
                question_raw = item.get("Question", "")
                correct_ans = item.get("Correct Answer", "")
                inc_ans_1 = item.get("Incorrect Answer 1", "")
                inc_ans_2 = item.get("Incorrect Answer 2", "")
                inc_ans_3 = item.get("Incorrect Answer 3", "")
                
                options = [correct_ans, inc_ans_1, inc_ans_2, inc_ans_3]
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
                # === 分支 3: 普通填空题逻辑 (MATH/GSM8K) ===
                # 这里才去读之前探测到的列名
                q_text = item.get(q_col_test, "")
                a_text = item.get(a_col_test, "")

            # 统一写入
            f.write(json.dumps({
                "id": str(real_id),
                "question": q_text,
                "golden_answers": [str(a_text)]
            }) + "\n")
            
    return True