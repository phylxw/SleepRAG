import os
import json
from datasets import load_dataset

import os
import json
from datasets import load_dataset

def normalize_code_instance(item, dataset_type="mbpp"):
    """
    数据清洗与标准化 (保持原逻辑不变)
    """
    normalized = {}
    
    if dataset_type == "mbpp":
        # MBPP (sanitized) 
        question = item.get("prompt") or item.get("text")
        answer = item.get("code")
        
        normalized = {
            "id": str(item.get("task_id", "")),
            # 🔥 [核心修复] FlashRAG 强制要求 'contents' 字段
            "contents": question, 
            "question": question,
            "golden_answers": [answer] if answer else [],
            **item 
        }
        
    # ... (humaneval part if needed)
    
    return normalized

def prepare_mbpp(corpus_file, test_file, cfg, need_split):
    """
    MBPP 专用准备函数
    
    Args:
        corpus_file: 记忆库保存路径 (构建自 MBPP Train set)
        test_file: 测试集保存路径 (构建自 MBPP Validation 或 Test set)
        cfg: Hydra 配置对象
    """

    print(f"\n🔨 [Prepare] 进入 MBPP 数据准备流程...")

    # ==========================================
    # 2. 准备记忆库 (Corpus - 仅使用训练集)
    # ==========================================
    if not os.path.exists(corpus_file):
        print(f"📚 [Corpus] 正在构建 MBPP 记忆库 (Train Set Only)...")
        try:
            # ⚠️ 关键点：因为我们要测 MBPP，所以记忆库只能包含 train (可能包含 prompt split)，
            # 绝对不能包含 validation 和 test，否则就是数据泄漏。
            corpus_split = "train+prompt" 
            
            mbpp_corpus_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=corpus_split)
            
            os.makedirs(os.path.dirname(corpus_file), exist_ok=True)
            with open(corpus_file, 'w', encoding='utf-8') as f:
                for item in mbpp_corpus_ds:
                    processed = normalize_code_instance(item, dataset_type="mbpp")
                    f.write(json.dumps(processed, ensure_ascii=False) + "\n")
            
            print(f"    ✅ MBPP 记忆库已保存: {corpus_file} (来源于 {corpus_split}, 共 {len(mbpp_corpus_ds)} 条)")
        except Exception as e:
            print(f"    ❌ 加载 MBPP Corpus 失败: {e}")
            return False

    # ==========================================
    # 3. 准备测试集 (Validation 或 Test)
    # ==========================================
    # 获取目标 split 配置，默认为 test
    is_val = need_split
    
    
    try:
        # 1. 第一步：根据配置加载 MBPP 原生的 validation 或 test 分割
        # MBPP 原生支持: 'train', 'validation', 'test', 'prompt'
        if is_val:
            print(f"    🚀 模式: 验证集 (Validation Split)")
            candidate_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="validation")
        else:
            print(f"    🚀 模式: 最终测试 (Test Split)")
            candidate_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

        # 2. 第二步：应用 start_index 和 debug_num 进行二次切片 (保持原逻辑)
        p_start = int(cfg.parameters.get("start_index", 0) or 0)
        p_debug = cfg.parameters.get("debug_num") # 可能为 None
        
        candidate_len = len(candidate_ds)
        
        # 计算切片终点
        if p_debug:
            limit = int(p_debug)
            p_end = min(p_start + limit, candidate_len)
            print(f"    ✂️ [Debug Mode] 启用切片: Relative Index {p_start} -> {p_end} (共 {p_end - p_start} 条)")
        else:
            p_end = candidate_len
            print(f"    📊 [Full Mode] 全量模式: Relative Index {p_start} -> End ({candidate_len} 条)")
            
        # 异常检查
        if p_start >= candidate_len:
            print(f"    ⚠️ [Warning] start_index ({p_start}) 超出了当前数据集长度 ({candidate_len})，将生成空文件！")
            final_ds = []
        else:
            # 执行切片
            final_ds = candidate_ds.select(range(p_start, p_end))

        # 3. 保存文件
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in final_ds:
                processed = normalize_code_instance(item, dataset_type="mbpp")
                f.write(json.dumps(processed, ensure_ascii=False) + "\n")
        
        print(f"    ✅ MBPP 测试/验证集 已保存: {test_file} (最终写入 {len(final_ds)} 条)")
        
    except Exception as e:
        print(f"    ❌ 加载 MBPP Test/Val 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True