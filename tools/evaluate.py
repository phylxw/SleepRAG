
import os
import re
import json
from utils.math_reward import last_boxed_only_string, remove_boxed

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))
from math_verify import parse, verify

def judge_math_item(item):
    """
    使用 math_verify 进行解析与比较
    """
    # 获取原始数据
    pred_raw = item.pred if hasattr(item, 'pred') else item.get('pred', "")
    golden_answers = item.golden_answers if hasattr(item, 'golden_answers') else item.get('golden_answers', [])
    gold_raw = golden_answers[0] if golden_answers else ""

    # 1. 解析 Golden Answer
    gold_parsed = parse(str(gold_raw))
    
    # 🔥 [新增修复逻辑]
    # 如果 parse 出来是空的 (因为 HMMT 这种数据集答案没带 boxed)，
    # 我们就手动给它套上 \boxed{} 再解析一次，强制让 math_verify 认出它。
    if not gold_parsed:
        gold_parsed = parse(f"\\boxed{{{str(gold_raw)}}}")

    # 2. 解析 Prediction (模型输出通常已经按 Prompt 要求带了 boxed，所以一般没事)
    pred_parsed = parse(str(pred_raw))

    # 3. 使用 verify 比较
    try:
        is_right = verify(gold_parsed, pred_parsed)
    except Exception:
        is_right = False

    return is_right, str(gold_parsed), str(pred_parsed)

def evaluate_results(results, experiment_name, result_log_file):
    correct = 0
    total = len(results)
    
    # 确保目录存在 
    os.makedirs(os.path.dirname(result_log_file), exist_ok=True)

    with open(result_log_file, "a", encoding="utf-8") as f:
        header = f"\n{'='*20} {experiment_name} {'='*20}\n"
        print(header.strip()) 
        f.write(header)
        
        for i, item in enumerate(results):
            # 获取题目用于展示 [cite: 18]
            question = item.question if hasattr(item, 'question') else item.get('question', "")
            pred_raw = item.pred if hasattr(item, 'pred') else item.get('pred', "")

            # 核心判断逻辑
            is_right, gold_val, pred_val = judge_math_item(item)
            if is_right: 
                correct += 1

            # 日志记录：记录解析前后的对比 [cite: 19, 20]
            log_entry = (
                f"\n[ID]: {i}\n"
                f"[Question]: {str(question)}...\n"
                f"[Gold Parsed]: {gold_val}\n"
                f"[Pred Parsed]: {pred_val}\n"
                f"[Pred All]: {pred_raw}\n"
                f"[Result]: {'✅ Correct' if is_right else '❌ Wrong'}\n"
                f"{'-'*30}\n"
            )
            log_print = (
                f"\n[ID]: {i}\n"
                f"[Question]: {str(question)}...\n"
                f"[Gold Parsed]: {gold_val}\n"
                f"[Pred Parsed]: {pred_val}\n"
                f"[Result]: {'✅ Correct' if is_right else '❌ Wrong'}\n"
                f"{'-'*30}\n"
            )
            f.write(log_entry)
            
            # 控制台只打印前 5 条预览 [cite: 20]
            if i < 5: 
                print(log_print.strip())

        # 统计最终准确率 [cite: 21]
        acc = correct / total * 100 if total > 0 else 0
        summary = (
            f"\n📊 统计 ({experiment_name}):\n"
            f"Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}%\n"
            f"{'='*50}\n"
        )
        print(summary)
        f.write(summary)
        
    return acc
