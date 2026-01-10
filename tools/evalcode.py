import requests
import re
from concurrent.futures import ThreadPoolExecutor
import os

class CodeEvaluator:
    def __init__(self, server_url="http://localhost:8080", max_workers=16):
        self.server_url = server_url
        self.max_workers = max_workers

    def extract_python_code(self, text: str) -> str:
        """从 Markdown 中提取 Python 代码"""
        pattern = r"```python(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # 🔥 核心修正：只去除首尾换行，保留缩进
            return match.group(1).strip('\n')
        return text.strip('\n')

    def evaluate_one(self, dataset_type, pred_str, task_data):
        """
        单条评测逻辑
        """
        code_body = self.extract_python_code(pred_str)
        full_code = ""
        
        if dataset_type == 'humaneval':
            # HumanEval: Prompt + Code + Test
            entry_point = task_data.get('entry_point', 'candidate')
            test_code = task_data.get('test', '')
            full_code = f"{task_data['prompt']}\n{code_body}\n\n{test_code}"

        elif dataset_type == 'mbpp':
            # MBPP: Setup + Code + Test List
            # 注意兼容性：有些数据可能没有 test_setup_code
            test_list = task_data.get('test_list', [])
            setup_code = task_data.get('test_setup_code', "")
            tests_str = "\n".join(test_list)
            full_code = f"{setup_code}\n{code_body}\n\n{tests_str}"
            
        else:
            return 0.0

        try:
            resp = requests.post(f"{self.server_url}/run_code", json={
                'code': full_code,
                'language': 'python'
            }, timeout=10) # 建议保持 10s 配合服务端限制
            
            if resp.status_code == 200:
                res_json = resp.json()
                if res_json.get('status') == 'Success':
                    return 1.0
            return 0.0
        except Exception as e:
            # 生产环境可以选择打印 log 或忽略
            # print(f"⚠️ [Eval Error] {e}") 
            return 0.0

    def evaluate_batch(self, dataset_type, pred_list, task_data_list):
        """
        ⚡ 批量并发评测 (供 sglang 模式使用)
        """
        print(f"⚖️ [CodeEval] 正在并发评测 {len(pred_list)} 条代码 (Workers={self.max_workers})...")
        
        scores = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for pred, item in zip(pred_list, task_data_list):
                futures.append(executor.submit(self.evaluate_one, dataset_type, pred, item))
            
            for future in futures:
                scores.append(future.result())
                
        return scores

def evaluate_code_results(results, experiment_name, result_log_file, dataset_type="humaneval", server_url="http://localhost:8080"):
    """
    🔥 新增函数：专门用于评测代码 (HumanEval/MBPP)
    特点：
    1. 内部使用 CodeEvaluator.evaluate_batch 实现并发加速
    2. 日志格式与数学评测完全保持一致
    """
    # 1. 初始化评测器
    evaluator = CodeEvaluator(server_url=server_url, max_workers=16) # 这里的 workers 可以根据需要调整
    
    # 2. 准备数据 (批量提取)
    # 注意：CodeEvaluator 需要 raw data (dict 形式) 来获取 test_list 等字段
    # 如果 results 是 FlashRAG 的对象，通常可以将 item 转为 dict 或直接传
    task_data_list = []
    preds_list = []
    
    for item in results:
        # 兼容性处理：如果是对象则转 dict，如果是 dict 则直接用
        data_dict = item.__dict__ if hasattr(item, '__dict__') else item
        task_data_list.append(data_dict)
        
        # 获取预测值
        pred = item.pred if hasattr(item, 'pred') else item.get('pred', "")
        preds_list.append(pred)

    # 3. 🔥 并发批量评测 (这是速度的关键)
    print(f"🚀 [Eval] 正在并发评测 {len(results)} 条代码数据 ({dataset_type})...")
    scores = evaluator.evaluate_batch(dataset_type, preds_list, task_data_list)
    
    # 4. 统计与日志记录 (模仿 evaluate_math_results 的风格)
    correct = 0
    total = len(results)
    
    os.makedirs(os.path.dirname(result_log_file), exist_ok=True)

    with open(result_log_file, "a", encoding="utf-8") as f:
        header = f"\n{'='*20} {experiment_name} (Code) {'='*20}\n"
        print(header.strip()) 
        f.write(header)
        
        for i, (item, score, pred) in enumerate(zip(results, scores, preds_list)):
            is_right = (score == 1.0)
            if is_right:
                correct += 1
            
            # 获取问题用于展示 (Code 任务通常 prompt 就是 question)
            # 尝试获取 prompt, text 或 question 字段
            q_text = ""
            if hasattr(item, 'prompt'): q_text = item.prompt
            elif hasattr(item, 'text'): q_text = item.text # MBPP
            elif isinstance(item, dict): q_text = item.get('prompt', item.get('text', ""))
            
            # 为了日志好看，提取纯代码部分展示
            extracted_code = evaluator.extract_python_code(pred)
            
            # [cite_start]日志记录 [cite: 19, 20]
            log_entry = (
                f"\n[ID]: {i}\n"
                f"[Question/Prompt]: {str(q_text)[:80]}...\n" # 防止 Prompt 太长
                f"[Pred Extracted]: \n{extracted_code[:200]}...\n" # 只展示代码前200字符
                f"[Result]: {'✅ Correct' if is_right else '❌ Wrong (Pass@1)'}\n"
                f"{'-'*30}\n"
            )
            f.write(log_entry)
            
            # 控制台预览前 3 条
            if i < 3:
                print(log_entry.strip())

        # [cite_start]统计最终准确率 [cite: 21]
        acc = correct / total * 100 if total > 0 else 0
        summary = (
            f"\n📊 统计 ({experiment_name}):\n"
            f"Dataset: {dataset_type.upper()}\n"
            f"Total: {total}, Correct: {correct}, Accuracy (Pass@1): {acc:.2f}%\n"
            f"{'='*50}\n"
        )
        print(summary)
        f.write(summary)

    return acc

# 简单的自测入口，确认文件没拷错
if __name__ == "__main__":
    print("🚀 [Self-Test] 开始调试...")
    evaluator = CodeEvaluator(server_url="http://localhost:8080")

    # 1. HumanEval 测试
    he_task_data = {
        "prompt": "def multiply(a, b):",
        "entry_point": "multiply",
        "test": "assert multiply(2, 3) == 6"
    }
    # 注意：这里的 return 前面有4个空格
    he_pred = """```python
    return a * b
```"""
    
    print("\n-------------------------------------")
    print("🧪 测试场景 1: HumanEval")
    score_he = evaluator.evaluate_one("humaneval", he_pred, he_task_data)
    print(f"➡️ 结果: {score_he}")

    # 2. MBPP 测试
    print("\n-------------------------------------")
    print("🧪 测试场景 2: MBPP")
    mbpp_task_data = {
        "test_setup_code": "import math",
        "test_list": ["assert get_sqrt(4) == 2.0"]
    }
    mbpp_pred = "def get_sqrt(n): return math.sqrt(n)"
    score_mbpp = evaluator.evaluate_one("mbpp", mbpp_pred, mbpp_task_data)
    print(f"➡️ 结果: {score_mbpp}")