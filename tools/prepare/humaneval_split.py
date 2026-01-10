import os
import json
from datasets import load_dataset

def normalize_code_instance(item, dataset_type="humaneval"):
    """
    数据清洗与标准化
    🔥 核心目标：保留原始字段，并满足 FlashRAG 对 contents 字段的要求
    """
    normalized = {}
    
    if dataset_type == "mbpp":
        # MBPP (sanitized) 
        question = item.get("prompt") or item.get("text")
        answer = item.get("code")
        
        normalized = {
            "id": str(item.get("task_id", "")),
            
            # 🔥 [核心修复] FlashRAG 强制要求 'contents' 字段用于检索和语言检测
            "contents": question, 
            
            "question": question,
            "golden_answers": [answer] if answer else [],
            **item # 保留原始字段 (test_list 等)
        }
        
    elif dataset_type == "humaneval":
        # HumanEval
        question = item.get("prompt")
        answer = item.get("canonical_solution")
        
        normalized = {
            "id": str(item.get("task_id", "")),
            
            # 🔥 [核心修复] 同上，加上 contents
            "contents": question,
            
            "question": question,
            "golden_answers": [answer] if answer else [],
            **item # 保留 entry_point 等
        }
    
    return normalized

def prepare_humaneval(corpus_file, test_file, cfg, need_split):
    """
    HumanEval 专用准备函数
    
    Args:
        corpus_file: 记忆库保存路径 (应为 mbpp)
        test_file: 测试集保存路径 (应为 humaneval split)
        cfg: Hydra 配置对象
        need_split: (在此任务中暂不用于 Corpus 切分，主要逻辑由 cfg 控制)
    """

    print(f"\n🔨 [Prepare] 进入 Code Generation 数据准备流程...")

    # ==========================================
    # 2. 准备记忆库 (MBPP Sanitized)
    # ==========================================
    if not os.path.exists(corpus_file):
        print(f"📚 [Corpus] 正在构建 MBPP (sanitized) 记忆库...")
        try:
            # 使用 sanitized 版本，质量更高，适合做 RAG 底座
            mbpp_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
            
            os.makedirs(os.path.dirname(corpus_file), exist_ok=True)
            with open(corpus_file, 'w', encoding='utf-8') as f:
                for item in mbpp_ds:
                    processed = normalize_code_instance(item, dataset_type="mbpp")
                    f.write(json.dumps(processed, ensure_ascii=False) + "\n")
            
            print(f"    ✅ MBPP 记忆库已保存: {corpus_file} ({len(mbpp_ds)} 条)")
        except Exception as e:
            print(f"    ❌ 加载 MBPP 失败: {e}")
            return False


    # ==========================================
    # 3. 准备测试集 (HumanEval Split)
    # ==========================================
    print(f"🧪 [Test] 正在构建 HumanEval 测试集...")
    try:
        he_ds = load_dataset("openai_humaneval", split="test") # HumanEval 只有 test split (164条)
        total_len = len(he_ds)
        mid_point = total_len // 2 # 82
        
        # 读取配置中的 split 意图
        # 默认为 "test" (后半部分)，如果是 "validation" 则取前半部分
        target_split = cfg.experiment.get("test_split", "test")
        
        if target_split == "validation":
            print(f"    🚀 模式: 验证集 (Validation)")
            print(f"    ✂️ 切分: 前 {mid_point} 题 (Index 0-{mid_point-1})")
            selected_ds = he_ds.select(range(0, mid_point))
        else:
            print(f"    🚀 模式: 最终测试 (Test)")
            print(f"    ✂️ 切分: 后 {total_len - mid_point} 题 (Index {mid_point}-{total_len-1})")
            selected_ds = he_ds.select(range(mid_point, total_len))

        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in selected_ds:
                processed = normalize_code_instance(item, dataset_type="humaneval")
                f.write(json.dumps(processed, ensure_ascii=False) + "\n")
        
        print(f"    ✅ HumanEval ({target_split}) 已保存: {test_file} ({len(selected_ds)} 条)")
        
    except Exception as e:
        print(f"    ❌ 加载 HumanEval 失败: {e}")
        return False

    return True

