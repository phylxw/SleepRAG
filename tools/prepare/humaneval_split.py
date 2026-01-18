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
    is_val = need_split
    # ==========================================
    # 2. 准备记忆库 (MBPP Sanitized)
    # ==========================================
    if not os.path.exists(corpus_file):
        print(f"📚 [Corpus] 正在构建 MBPP (sanitized) 记忆库...")
        try:
            # 使用 sanitized 版本，质量更高，适合做 RAG 底座
            target_split = cfg.experiment.get("corpus_split", "train+validation+test+prompt")
            mbpp_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=target_split)
            
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
    try:
        he_ds = load_dataset("openai_humaneval", split="test") # HumanEval 只有 test split (164条)
        total_len = len(he_ds)
        mid_point = total_len // 2 # 82
        
        if is_val:
            print(f"    🚀 模式: 验证集 (Validation)")
            print(f"    ✂️ 原始范围: 前 {mid_point} 题 (Index 0-{mid_point-1})")
            # 选出前一半
            candidate_ds = he_ds.select(range(0, mid_point))
        else:
            print(f"    🚀 模式: 最终测试 (Test)")
            print(f"    ✂️ 原始范围: 后 {total_len - mid_point} 题 (Index {mid_point}-{total_len-1})")
            # 选出后一半
            candidate_ds = he_ds.select(range(mid_point, total_len))

        # 2. 第二步：应用 start_index 和 debug_num 进行二次切片
        # 获取参数，带默认值处理
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
            
        # 异常检查：如果 start 超过了长度
        if p_start >= candidate_len:
            print(f"    ⚠️ [Warning] start_index ({p_start}) 超出了当前数据集长度 ({candidate_len})，将生成空文件！")
            final_ds = []
        else:
            # 执行切片 (注意：这里的 range 是相对于 candidate_ds 的 0 开始的)
            final_ds = candidate_ds.select(range(p_start, p_end))

        # 3. 保存文件
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in final_ds:
                processed = normalize_code_instance(item, dataset_type="humaneval")
                f.write(json.dumps(processed, ensure_ascii=False) + "\n")
        
        print(f"    ✅ HumanEval 测试/验证集 已保存: {test_file} (最终写入 {len(final_ds)} 条)")
        
    except Exception as e:
        print(f"    ❌ 加载 HumanEval 失败: {e}")
        # 为了调试方便，打印完整的错误堆栈
        import traceback
        traceback.print_exc()
        return False

    return True

