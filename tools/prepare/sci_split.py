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

def prepare_sciknow(corpus_path: str, test_path: str, cfg: DictConfig , need_split) -> bool:
    is_val = need_split
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

    # =========================================================
    # 🔥 核心修改：双层切分逻辑
    # =========================================================
    
    # --- Stage 1: 物理隔离 (80% 潜在记忆池 vs 20% 最终测试集) ---
    # 这是一成不变的，保证最终测试集 (final_test_pool) 永远不被污染
    split_idx_1 = int(len(final_data) * 0.8)
    corpus_pool = final_data[:split_idx_1]      # 80%
    final_test_pool = final_data[split_idx_1:]  # 20%
    
    print(f"   📉 [Stage 1] 物理隔离: 潜在记忆池 {len(corpus_pool)} 条 | 最终保留测试集 {len(final_test_pool)} 条")

    # --- Stage 2: 根据 is_val 决定实际使用的据 ---
    if is_val:
        # ✅ 验证/优化模式：
        # 从 80% 的 corpus_pool 里，再切分出验证集 (默认 10%)
        # 剩下的 90% 做记忆。final_test_pool 在这里不使用。
        split_ratio = cfg.parameters.get("split_ratio", 0.9)
        split_idx_2 = int(len(corpus_pool) * split_ratio)
        
        real_corpus_data = corpus_pool[:split_idx_2]      # 实际写入记忆库的
        target_test_data = corpus_pool[split_idx_2:]      # 实际写入测试文件(验证集)的
        
        print(f"   🔀 [Validation Mode] 启动验证模式:")
        print(f"     👉 从记忆池中划分 {len(target_test_data)} 条做验证 (Split Ratio: {split_ratio})")
        print(f"     👉 实际记忆库大小: {len(real_corpus_data)}")
        
    else:
        # 🚀 最终测试模式：
        # 记忆库使用完整的 corpus_pool (80%)
        # 测试集使用之前隔离好的 final_test_pool (20%)
        real_corpus_data = corpus_pool
        target_test_data = final_test_pool
        
        print(f"   🚀 [Final Test Mode] 启动最终测试模式:")
        print(f"     👉 使用完整的潜在记忆池 ({len(real_corpus_data)} 条)")
        print(f"     👉 使用预留的最终测试集 ({len(target_test_data)} 条)")

    # =========================================================
    # 写入流程 (使用 real_corpus_data 和 target_test_data)
    # =========================================================

    # 3. 写入 Memory
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        count = 0
        # 🔥 注意：这里遍历的是 real_corpus_data
        for i, item in enumerate(tqdm(real_corpus_data, desc="Writing Corpus")):
            q_text, a_text, is_valid = _format_sciknow_instance(item)
            if is_valid:
                content = f"Question: {q_text}\nAnswer: {a_text}"
                f.write(json.dumps({"id": str(count), "contents": content}) + "\n")
                count += 1
            
    # 4. 写入 Test (保留 Debug 切片逻辑)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    # 读取调试参数
    start_index = int(cfg.parameters.get("start_index", 0) or 0)
    debug_num = cfg.parameters.get("debug_num")
    
    # 对 target_test_data 进行切片处理
    if debug_num:
        limit = int(debug_num)
        end_idx = min(start_index + limit, len(target_test_data))
        test_data_slice = target_test_data[start_index : end_idx]
        print(f"   🐛 [Debug] 仅写入 {len(test_data_slice)} 条测试数据 (Start: {start_index})")
    else:
        test_data_slice = target_test_data[start_index:]
        print(f"   📊 [Full] 写入 {len(test_data_slice)} 条测试数据")

    with open(test_path, "w", encoding="utf-8") as f:
        count = 0 # 重置 ID
        for i, item in enumerate(tqdm(test_data_slice, desc="Writing Test")):
            q_text, a_text, is_valid = _format_sciknow_instance(item)
            if is_valid:
                f.write(json.dumps({
                    "id": str(count), 
                    "question": q_text,
                    "golden_answers": [a_text]
                }) + "\n")
                count += 1
            
    print("✅ SciKnowEval 处理完成！")
    return True