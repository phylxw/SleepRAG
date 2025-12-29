import json
import os
import random
from datasets import load_dataset

# 要合并的年份
DATASETS_TO_MERGE = [
    "MathArena/hmmt_feb_2023",
    "MathArena/hmmt_feb_2024",
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

def merge_hmmt(output_path, cfg):
    """
    🔥 核心修改：接收 output_path 和 cfg 参数
    """
    print(f"🚀 [Merge] 开始合并 {len(DATASETS_TO_MERGE)} 个 HMMT 数据集...")
    
    # 确保父目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    all_data = []
    
    for ds_name in DATASETS_TO_MERGE:
        print(f"   📥 Loading: {ds_name} ...")
        try:
            ds = load_dataset(ds_name, split="test") 
        except:
            try:
                ds = load_dataset(ds_name, split="train")
            except Exception as e:
                print(f"   ❌ 跳过 {ds_name}: {e}")
                continue
                
        for item in ds:
            processed = normalize_instance(item)
            if processed['question'] and processed['golden_answers']:
                all_data.append(processed)

    # 🔥 [关键] 固定随机种子，保证每次 Shuffle 结果一致
    # 这样你的 start_index=40 才有意义，否则每次都是不同的题
    # random.seed(42)
    # random.shuffle(all_data)
    
    # 🔥 [关键] 读取参数并切片
    start_index = int(cfg.parameters.get("start_index", 0) or 0)
    debug_num = cfg.parameters.get("debug_num")
    
    total_len = len(all_data)
    end_index = total_len
    
    if debug_num:
        limit = int(debug_num)
        end_index = min(start_index + limit, total_len)
        print(f"✂️ [Debug Mode] 启用切片: Index {start_index} -> {end_index}")
    else:
        print(f"📊 [Full Mode] 全量模式: Index {start_index} -> End")

    # 执行切片
    final_data = all_data[start_index : end_index]

    # 重标 ID (保持 ID 的连续性，方便 Debug)
    # 我们让 ID 反映真实的索引位置 (start_index + i)
    for idx, item in enumerate(final_data):
        real_id = start_index + idx
        item['id'] = str(real_id) 

    print(f"💾 保存合并数据 ({len(final_data)} 条) 至: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("✅ HMMT 合并及切片完成！")