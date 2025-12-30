import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from omegaconf import DictConfig
from collections import Counter # 引入计数器，看看数据分布

def _format_sciknow_instance(item):
    """
    智能解析 SciKnowEval 的各种奇葩格式
    """
    question_raw = item.get("question", "").strip()
    choices = item.get("choices", None)
    answer_raw = item.get("answer", "")
    type_str = item.get("type", "")
    
    # 兼容数字答案 0 -> 'A'
    if isinstance(answer_raw, int):
        labels = ["A", "B", "C", "D", "E", "F"]
        if 0 <= answer_raw < len(labels):
            answer_raw = labels[answer_raw]
        else:
            answer_raw = str(answer_raw) # 兜底

    # 1. 必杀技：严格检查空值 (修复 0 被误杀的 bug)
    if answer_raw is None or str(answer_raw).strip() == "":
        return "", "", False

    if "true_or_false" in str(type_str).lower() and not choices:
        choices = ["True", "False"] 
        # 顺便把答案归一化：如果答案是 "True" -> 转成 "A", "False" -> "B"
        if str(answer_raw).lower() == "true": answer_raw = "A"
        if str(answer_raw).lower() == "false": answer_raw = "B"
    
    options_str = ""
    
    # === 情况 A: 标准字典格式 ===
    if isinstance(choices, dict) and "text" in choices:
        texts = choices["text"]
        labels = choices.get("label", [])
        if not labels: 
            labels = ["A", "B", "C", "D", "E", "F"][:len(texts)]
        for l, t in zip(labels, texts):
            options_str += f"\n({l}) {t}"

    # === 情况 B: 列表格式 ===
    elif isinstance(choices, list):
        labels = ["A", "B", "C", "D", "E", "F"]
        for idx, text in enumerate(choices):
            label = labels[idx] if idx < len(labels) else str(idx)
            options_str += f"\n({label}) {text}"
            
    # === 情况 C: 字符串 ===
    elif isinstance(choices, str):
        options_str = f"\n{choices}"

    q_text = question_raw + options_str
    a_text = str(answer_raw).strip()
    
    return q_text, a_text, True

def prepare_sciknow(corpus_path: str, test_path: str, cfg: DictConfig) -> bool:
    
    # if os.path.exists(corpus_path) and os.path.exists(test_path):
    #     print(f"✅ [SciKnow] 检测到现有的 Corpus 和 Test 文件 (跳过切分)")
    #     return True

    print(f"⚡ [Auto-Split] 正在下载 SciKnowEval...")
    try:
        ds = load_dataset("hicai-zju/SciKnowEval", split="test") 
    except Exception as e:
        print(f"❌ SciKnowEval 下载失败: {e}")
        return False
    
    raw_data = list(ds)
    print(f"   📊 原始数据全量: {len(raw_data)}")

    # =========================================================
    # 🔍 上帝视角：扫描分布
    # =========================================================
    print("\n🧐 [Debug] 正在扫描数据分布...")
    domain_counter = Counter()
    type_counter = Counter()
    valid_candidates = []
    
    for item in tqdm(raw_data, desc="Scanning"):
        ans = item.get("answer")
        if ans is None or str(ans).strip() == "": continue
            
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
    # 🔥 过滤逻辑 (Domain + True/False)
    # =========================================================
    target_domain = cfg.experiment.get("target_domain")
    print(f"\n   🧹 清洗数据 (Domain: {target_domain} | Type: True/False Only)...")
    
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

        # 2. 题型过滤 (只保留 True/False)
        t = item.get("type", "")
        if "true_or_false" not in str(t).lower():
            skipped_type += 1
            continue
            
        final_data.append(item)

    print(f"   🚫 过滤统计: 领域不符={skipped_domain} | 非判断题={skipped_type}")
    print(f"   ✅ 有效数据: {len(final_data)} 条")

    if len(final_data) == 0:
        print(f"❌ 错误: 筛选后数据为 0。请放宽条件。")
        return False
        
    # =========================================================
    # 处理流程 (打乱 -> 截断 -> 切分 -> 写入)
    # =========================================================
    
    random.seed(42) 
    random.shuffle(final_data)
    
    # 1. 总量截断 (total_limit)
    total_limit = cfg.experiment.get("total_limit")
    if total_limit:
        limit_val = int(total_limit)
        if limit_val < len(final_data):
            print(f"   ✂️ [Total Limit] 截取前 {limit_val} 条用于实验")
            final_data = final_data[:limit_val]

    # 2. 80/20 切分
    split_idx = int(len(final_data) * 0.8)
    corpus_data = final_data[:split_idx]
    test_data = final_data[split_idx:]
    
    print(f"   📉 切分结果: Memory库 {len(corpus_data)} 条 | Test集 {len(test_data)} 条")
    
    # 3. 写入 Memory
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        count = 0
        for i, item in enumerate(tqdm(corpus_data, desc="Writing Corpus")):
            q_text, a_text, is_valid = _format_sciknow_instance(item)
            if is_valid:
                content = f"Question: {q_text}\nAnswer: {a_text}"
                f.write(json.dumps({"id": str(count), "contents": content}) + "\n")
                count += 1
            
    # 4. 写入 Test (🔥 补上 Debug 切片逻辑)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    # 🔥🔥🔥 [新增] 读取 debug_num 和 start_index
    start_index = int(cfg.parameters.get("start_index", 0) or 0)
    debug_num = cfg.parameters.get("debug_num")
    
    if debug_num:
        limit = int(debug_num)
        end_idx = min(start_index + limit, len(test_data))
        # 对 test_data 进行切片，只写入这一小部分
        test_data_slice = test_data[start_index : end_idx]
        print(f"   🐛 [Debug Mode] Test集切片: 仅写入 {len(test_data_slice)} 条 (Start: {start_index})")
    else:
        # 如果没开 debug，就写全量 (从 start_index 开始到最后，或者全量)
        test_data_slice = test_data[start_index:]
        print(f"   📊 [Full Mode] 写入 Test集: {len(test_data_slice)} 条")

    with open(test_path, "w", encoding="utf-8") as f:
        count = 0 # 重置 ID
        for i, item in enumerate(tqdm(test_data_slice, desc="Writing Test")):
            q_text, a_text, is_valid = _format_sciknow_instance(item)
            if is_valid:
                f.write(json.dumps({
                    "id": str(count), # ID 从 0 开始
                    "question": q_text,
                    "golden_answers": [a_text]
                }) + "\n")
                count += 1
            
    print("✅ SciKnowEval 处理完成！")
    return True