import json
import os
import random
from datasets import load_dataset

# === 🛠️ 配置两组数据集 ===
# 验证集模式 (Optimization Phase): 使用过去两年的真题
HMMT_VAL_SETS = [
    "MathArena/hmmt_feb_2023",
    "MathArena/hmmt_feb_2024"
]

# 测试集模式 (Final Evaluation): 使用最新的真题 (完全隔离)
HMMT_TEST_SETS = [
    "MathArena/hmmt_feb_2025"
]

def normalize_instance(item):
    """统一格式: problem -> question, solution -> golden_answers"""
    question = item.get("problem") or item.get("question")
    answer = item.get("solution") or item.get("answer")
    if answer: answer = str(answer).strip()
    
    return {
        "id":  None,
        "question": question,
        "golden_answers": [answer] if answer else []
    }

def merge_hmmt(output_path, cfg,is_val):
    """
    is_val是True时代表是验证，is_val是False时代表是最终测试
    """
    
    if is_val == False:
        target_datasets = HMMT_TEST_SETS
        print(f"🚀 [HMMT] 启动最终测试模式 (Final Test)")
        print(f"    🎯 目标年份: 2025")
    else:
        target_datasets = HMMT_VAL_SETS
        print(f"🚀 [HMMT] 启动验证/优化模式 (Validation)")
        print(f"    🎯 目标年份: 2023 + 2024")

    # 确保父目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    all_data = []
    
    # 2. 遍历加载对应列表的数据
    for ds_name in target_datasets:
        print(f"    📥 Loading: {ds_name} ...")
        try:
            ds = load_dataset(ds_name, split="test") 
        except:
            try:
                ds = load_dataset(ds_name, split="train")
            except Exception as e:
                print(f"    ❌ 跳过 {ds_name}: {e}")
                continue
                
        for item in ds:
            processed = normalize_instance(item)
            if processed['question'] and processed['golden_answers']:
                all_data.append(processed)

    # 3. [关键] 验证集需要 Shuffle 混合两年的题，测试集通常不需要
    # 为了保证实验可复现，这里建议开启 Shuffle 并固定 Seed
    if is_val:
        print("    🔀 [Shuffle] 正在混合 2023 和 2024 的题目...")
        random.seed(42)
        random.shuffle(all_data)
    
    # 4. 读取调试参数并切片
    start_index = int(cfg.parameters.get("start_index", 0) or 0)
    debug_num = cfg.parameters.get("debug_num")
    
    total_len = len(all_data)
    
    if debug_num:
        limit = int(debug_num)
        end_index = min(start_index + limit, total_len)
        print(f"✂️ [Debug Mode] 启用切片: Index {start_index} -> {end_index}")
    else:
        end_index = total_len
        print(f"📊 [Full Mode] 全量模式: Index {start_index} -> End ({total_len} 条)")

    # 执行切片
    final_data = all_data[start_index : end_index]

    # 5. 重标 ID (保持 ID 的连续性)
    for idx, item in enumerate(final_data):
        real_id = start_index + idx
        item['id'] = str(real_id) 

    print(f"💾 保存合并数据 ({len(final_data)} 条) 至: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("✅ HMMT 数据准备完成！")