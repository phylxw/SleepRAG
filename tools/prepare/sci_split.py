import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from omegaconf import DictConfig
from collections import Counter

def _format_sciknow_instance(item):
    """
    智能解析 SciKnowEval (MCQ 专用版 - 适配 answerKey)
    
    修改点：
    1. 优先读取 'answerKey'，如果为空才读 'answer'。
    2. 专门处理 answerKey 为数字索引 (0->A, 1->B) 的情况。
    """
    question_raw = item.get("question", "").strip()
    choices = item.get("choices", None)
    
    # === 🔥 修改点 1: 优先读取 answerKey ===
    # SciKnowEval 的选择题标准答案通常在这里
    answer_raw = item.get("answerKey")
    
    # 兜底：如果 answerKey 没东西，再回去读 answer
    if answer_raw is None or str(answer_raw).strip() == "":
        answer_raw = item.get("answer", "")

    # 1. 判空
    if answer_raw is None or str(answer_raw).strip() == "":
        return "", "", False

    # 2. 如果没有选项，那肯定不是选择题，直接跳过
    if not choices:
        return "", "", False

    # === 归一化处理 ===
    ans_str = str(answer_raw).strip()
    labels_pool = ["A", "B", "C", "D", "E", "F", "G", "H"]
    final_answer = ans_str 
    options_str = ""

    # === 处理选项列表 (List) ===
    if isinstance(choices, list):
        # 构造 (A) xxx (B) xxx
        for idx, text in enumerate(choices):
            label = labels_pool[idx] if idx < len(labels_pool) else str(idx)
            options_str += f"\n({label}) {text}"
            
            # 文本反向匹配 (防止 answerKey 给的是 "Carbon" 这种文本)
            if ans_str == str(text) or ans_str == str(text).strip():
                final_answer = label

        # === 🔥 修改点 2: 处理数字索引答案 ===
        # 情况 A: answerKey 是整数类型 (e.g., 0)
        if isinstance(answer_raw, int) and 0 <= answer_raw < len(choices):
            final_answer = labels_pool[answer_raw]
        
        # 情况 B: answerKey 是数字字符串 (e.g., "0")
        elif ans_str.isdigit():
            idx = int(ans_str)
            if 0 <= idx < len(choices):
                final_answer = labels_pool[idx]

    # === 处理选项字典 (Dict) ===
    # 格式通常是 {'text': ['a', 'b'], 'label': ['A', 'B']}
    elif isinstance(choices, dict) and "text" in choices:
        texts = choices["text"]
        labels = choices.get("label", labels_pool[:len(texts)])
        
        for l, t in zip(labels, texts):
            options_str += f"\n({l}) {t}"
            # 文本反向匹配
            if ans_str == str(t) or ans_str == str(t).strip():
                final_answer = l
                
    # === 兜底: 确保最终答案是 A/B/C/D 这样的字母 ===
    if len(str(final_answer)) == 1 and str(final_answer).upper() in labels_pool:
        final_answer = str(final_answer).upper()

    # 拼装
    q_text = question_raw + options_str
    
    return q_text, final_answer, True

def prepare_sciknow(corpus_path: str, test_path: str, cfg: DictConfig , need_split) -> bool:
    # 1. 检测记忆库是否已存在
    memory_exists = os.path.exists(corpus_path)
    if memory_exists:
        print(f"✅ [Cache] 记忆库文件已存在，将跳过生成步骤: {corpus_path}")
    else:
        print(f"⚠️ [Init] 记忆库缺失，准备生成...")
    
    is_val = need_split
    
    print(f"⚡ [Auto-Split] 正在下载 SciKnowEval...")
    try:
        ds = load_dataset("hicai-zju/SciKnowEval", split="test") 
    except Exception as e:
        print(f"❌ SciKnowEval 下载失败: {e}")
        return False
    
    raw_data = list(ds)
    print(f"   📊 原始数据全量: {len(raw_data)}")

    # =========================================================
    # 🔍 扫描数据分布 (修改处：修复判空逻辑)
    # =========================================================
    print("\n🧐 [Debug] 正在扫描数据分布...")
    domain_counter = Counter()
    type_counter = Counter()
    valid_candidates = []
    
    for item in tqdm(raw_data, desc="Scanning"):
        # 🔥 修改点 1: 只要 answer 或 answerKey 有一个不为空，就算有效数据
        ans = item.get("answer")
        ans_key = item.get("answerKey")
        
        has_ans = (ans is not None and str(ans).strip() != "")
        has_key = (ans_key is not None and str(ans_key).strip() != "")
        
        if not has_ans and not has_key:
            continue
            
        d = item.get("domain", "Unknown")
        if isinstance(d, list) and d: d = d[0]
        t = item.get("type", "Unknown")
        
        domain_counter[str(d)] += 1
        type_counter[str(t)] += 1
        valid_candidates.append(item)

    print(f"\n📈 [数据统计报告]")
    print(f"   👉 可用领域: {domain_counter}")
    print(f"   👉 可用题型: {type_counter}")
    
    if len(valid_candidates) == 0:
        print("\n❌ 错误: 数据集无有效答案样本。")
        return False

    # =========================================================
    # 🧹 清洗数据 (MCQ Only)
    # =========================================================
    target_domain = cfg.experiment.get("target_domain")
    print(f"\n   🧹 清洗数据 (Domain: {target_domain} | Type: MCQ Only)...")
    
    final_data = []
    skipped_domain = 0
    skipped_type = 0
    
    for item in valid_candidates:
        # 1. 领域过滤
        d = item.get("domain", "")
        if isinstance(d, list) and d: d = d[0]
        
        if target_domain and d != target_domain:
            skipped_domain += 1
            continue

        # 2. 题型过滤 (只留 MCQ)
        t = str(item.get("type", "")).lower()
        if "mcq" not in t and "multiple_choice" not in t:
            skipped_type += 1
            continue
            
        final_data.append(item)

    print(f"   🚫 过滤统计: 领域不符={skipped_domain} | 非 MCQ 题型={skipped_type}")
    print(f"   ✅ 有效数据: {len(final_data)} 条")

    if len(final_data) == 0:
        print(f"❌ 错误: 筛选后数据为 0。请放宽条件。")
        return False
        
    # =========================================================
    # 处理流程
    # =========================================================
    
    random.seed(42) 
    # random.shuffle(final_data)
    
    # 1. 总量截断
    total_limit = cfg.experiment.get("total_limit")
    if total_limit:
        limit_val = int(total_limit)
        if limit_val < len(final_data):
            print(f"   ✂️ [Total Limit] 截取前 {limit_val} 条用于实验")
            final_data = final_data[:limit_val]

    # --- Stage 1: 物理隔离 (80% 潜在记忆池 vs 20% 最终测试集) ---
    split_idx_1 = int(len(final_data) * 0.8)
    corpus_pool = final_data[:split_idx_1]      
    final_test_pool = final_data[split_idx_1:]  
    
    print(f"   📉 [Stage 1] 物理隔离: 潜在记忆池 {len(corpus_pool)} 条 | 最终保留测试集 {len(final_test_pool)} 条")

    # --- Stage 2: 根据 is_val 决定实际使用的数据 ---
    if is_val:
        split_ratio = cfg.parameters.get("split_ratio", 0.9)
        split_idx_2 = int(len(corpus_pool) * split_ratio)
        
        real_corpus_data = corpus_pool[:split_idx_2]      # 写入 memory.jsonl
        target_test_data = corpus_pool[split_idx_2:]      # 写入 test.jsonl (做验证)
        
        print(f"   🔀 [Validation Mode] 验证模式: 记忆库 {len(real_corpus_data)} | 验证集 {len(target_test_data)}")
        
    else:
        real_corpus_data = corpus_pool
        target_test_data = final_test_pool
        
        print(f"   🚀 [Final Test Mode] 测试模式: 记忆库 {len(real_corpus_data)} | 测试集 {len(target_test_data)}")

    # 3. 写入 Memory
    # 【修改点 3】增加条件判断：如果文件不存在才写
    if not memory_exists:
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        with open(corpus_path, "w", encoding="utf-8") as f:
            count = 0
            for i, item in enumerate(tqdm(real_corpus_data, desc="Writing Corpus")):
                q_text, a_text, is_valid = _format_sciknow_instance(item)
                if is_valid:
                    content = f"Question: {q_text}\nAnswer: {a_text}"
                    f.write(json.dumps({"id": str(count), "contents": content}, ensure_ascii=False) + "\n")
                    count += 1
        print(f"   💾 记忆库已生成: {count} 条")
    else:
        print(f"   ⏩ 记忆库已存在，跳过写入。")
            
    # 4. 写入 Test
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    start_index = int(cfg.parameters.get("start_index", 0) or 0)
    debug_num = cfg.parameters.get("debug_num")
    
    if debug_num:
        limit = int(debug_num)
        end_idx = min(start_index + limit, len(target_test_data))
        test_data_slice = target_test_data[start_index : end_idx]
        print(f"   🐛 [Debug] 仅写入 {len(test_data_slice)} 条测试数据")
    else:
        test_data_slice = target_test_data[start_index:]
        print(f"   📊 [Full] 写入 {len(test_data_slice)} 条测试数据")

    with open(test_path, "w", encoding="utf-8") as f:
        count = 0 
        for i, item in enumerate(tqdm(test_data_slice, desc="Writing Test")):
            q_text, a_text, is_valid = _format_sciknow_instance(item)
            if is_valid:
                f.write(json.dumps({
                    "id": str(count), 
                    "question": q_text,
                    "golden_answers": [a_text]
                }, ensure_ascii=False) + "\n")
                count += 1
            
    print("✅ SciKnowEval (判断题+选择题) 处理完成！")
    return True