import subprocess
import sys
import os
import time
from datetime import datetime

def run_step(script_name, timestamp, step_description, env=None):
    """
    运行单个脚本步骤
    :param script_name: 脚本文件名 (e.g., "pre.py")
    :param timestamp: 统一的时间戳 (e.g., "20251220_120000")
    :param step_description: 步骤描述
    :param env: 环境变量 (可选，用于指定 GPU)
    """
    print(f"\n{'='*60}")
    print(f"🚀 [Step: {step_description}] 正在启动 {script_name}...")
    print(f"🕒 由于是流水线作业，强制锁定时间戳: timestamp='{timestamp}'")
    print(f"{'='*60}\n")

    # 构造命令: python xxx.py timestamp="yyy"
    # 这样会覆盖 config.yaml 里的 ${now:...}，确保全程读写同一套文件
    cmd = [sys.executable, script_name, f"timestamp={timestamp}"]
    
    # 继承当前环境变量，如果有传入特定 env 则更新
    current_env = os.environ.copy()
    if env:
        current_env.update(env)

    # 启动子进程
    start_time = time.time()
    try:
        # check=True 表示如果脚本报错(退出码非0)，会直接抛出异常终止后续步骤
        subprocess.run(cmd, env=current_env, check=True)
    except subprocess.CalledProcessError:
        print(f"\n❌ [Error] {script_name} 运行失败！流水线已终止。")
        print("💡 请检查上方的错误日志。")
        sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"\n✅ [Success] {script_name} 完成 (耗时: {elapsed:.2f}s)")

def main():
    # 1. 生成本次流水线的唯一 ID (时间戳)
    # 这个时间戳会被传给所有脚本，确保它们读取的是同一批文件
    pipeline_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"🎬 开始执行全流程任务 | Pipeline ID: {pipeline_timestamp}")
    print(f"📂 工作目录: {os.getcwd()}")
    
    # 2. 显卡分配策略 (根据你的实际情况调整)
    # SGLang Server 应该已经在另外一个终端跑在 GPU 1,3,4,5 上了
    # 这里我们给客户端脚本分配剩下的 GPU (比如 0,2,6,7) 或者直接用某一张卡
    
    # 方案 A: 自动继承 (你在运行 main.py 时指定的 CUDA_VISIBLE_DEVICES)
    client_env = None 
    
    # 方案 B: 强制指定 (例如用 7 号卡跑 Embedding)
    # client_env = {"CUDA_VISIBLE_DEVICES": "7"} 

    # -----------------------------------------------------------
    # [步骤 1] Pre: 生成频次统计 & 初始语料
    # -----------------------------------------------------------
    run_step("pre.py", pipeline_timestamp, "1. 数据准备与频次统计", env=client_env)

    # -----------------------------------------------------------
    # [步骤 2] Cluster: 聚类 (需要 Embedding)
    # -----------------------------------------------------------
    # 注意: 如果你的 cluster.py 需要用 GPU 跑 Embedding，确保 client_env 里有卡
    run_step("cluster.py", pipeline_timestamp, "2. 题目自动聚类", env=client_env)

    # -----------------------------------------------------------
    # [步骤 3] Optimizer: 优化记忆 (高频聚合 + 低频扩写)
    # -----------------------------------------------------------
    run_step("optimizer.py", pipeline_timestamp, "3. 记忆库优化 (聚合/扩写)", env=client_env)

    # -----------------------------------------------------------
    # [步骤 4] Eval: 最终评测
    # -----------------------------------------------------------
    run_step("eval.py", pipeline_timestamp, "4. 最终 RAG 效果评测", env=client_env)

    print(f"\n🎉🎉🎉 全流程执行完毕！所有结果已生成。ID: {pipeline_timestamp}")

if __name__ == "__main__":
    main()