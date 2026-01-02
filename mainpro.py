import subprocess
import sys
import os
import time
from datetime import datetime
import hydra
from omegaconf import DictConfig
from utils.logger import setup_logging,Logger

def run_step(script_name, step_desc, overrides, env=None):
    print(f"\n{'='*80}")
    print(f"🚀 [Step: {step_desc}] 启动 {script_name}...")
    
    cmd = [sys.executable, script_name]
    
    print(f"📝 参数覆盖 (Overrides):")
    for key, value in overrides.items():
        final_value = value
        if isinstance(value, str) and (os.path.exists(os.path.dirname(value)) or os.path.isabs(value)):
             final_value = os.path.abspath(value)
        
        # 使用 ++ 强制覆盖/添加
        cmd.append(f"++{key}={final_value}") 
        print(f"   - {key} = {final_value}")
        
    print(f"{'-'*80}")

    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    
    # 强制让子进程的输出不缓冲，实时打到我们的 Logger 里
    current_env["PYTHONUNBUFFERED"] = "1"

    start_time = time.time()
    try:
        # 注意：这里不能用 capture_output=True，否则 Logger 抓不到子进程的实时输出
        # 我们直接让子进程继承 stdout，这样它的输出就会流向我们的 Logger
        subprocess.run(cmd, env=current_env, check=True)
    except subprocess.CalledProcessError:
        print(f"\n❌ [Error] {script_name} 运行失败！流水线已终止。")
        sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"✅ [Success] {script_name} 完成 (耗时: {elapsed:.2f}s)")

def get_round_paths(root_dir, pipeline_id, round_idx, tag="sci"):
    """
    定义每一轮所有的文件路径槽位
    """
    base_dir = os.path.join(root_dir, "results", pipeline_id, f"round_{round_idx}")
    os.makedirs(base_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return {
        "dir": base_dir,
        
        # --- 1. 核心记忆库文件 ---
        "corpus": os.path.join(base_dir, f"{tag}_corpus.jsonl"), # Round 0 初始
        "optimized_memory": os.path.join(base_dir, f"{tag}_optimized_memory_topk.jsonl"), # Round N 产出
        "test": os.path.join(base_dir, f"{tag}_test.jsonl"),

        # --- 2. 统计文件 (Stats) ---
        "stats": os.path.join(base_dir, f"{tag}_memory_stats.json"), # 初始/输入状态
        "stats_optimized": os.path.join(base_dir, f"{tag}_memory_optimized_stats.json"), # Opt重置后
        "stats_after": os.path.join(base_dir, f"{tag}_memory_after_stats.json"), # 🔥 Eval跑完后的最终状态 (给下一轮用)

        # --- 3. 频次文件 (Freq) ---
        "freq": os.path.join(base_dir, f"{tag}_memory_freq.jsonl"), # 初始/输入状态
        "freq_after": os.path.join(base_dir, f"{tag}_memory_after_freq.jsonl"), # 🔥 Eval跑完后的最终状态 (给下一轮用)

        # --- 4. 聚类中间产物 ---
        "cluster_output": os.path.join(base_dir, f"{tag}_clustered_result.jsonl"),
        "cluster_summary": os.path.join(base_dir, f"{tag}_cluster_summary.jsonl"),
        "cluster_vis": os.path.join(base_dir, f"{tag}_visualization.png"),
        "cluster_plot": os.path.join(base_dir, f"{tag}_cluster_distribution.png"),
        
        # --- 5. RAG 缓存 ---
        "rag_cache": os.path.join(root_dir, "rag_result_cache", pipeline_id, f"round_{round_idx}")
    }

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 0. 读取配置
    EXP_TAG = cfg.experiment.get("tag", "experiment")
    TOTAL_ROUNDS = cfg.parameters.get("total_rounds", 2)
    
    # 路径修正
    root_dir = cfg.paths.root if "paths" in cfg and "root" in cfg.paths else os.getcwd()
    root_dir = os.path.abspath(root_dir)

    # 1. 初始化
    pipeline_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{EXP_TAG}_Loop"
    setup_logging(root_dir, pipeline_timestamp) # 🔥 开启全局日志记录
    print(f"\n🎬 [Pipeline Start] 多轮迭代任务 | ID: {pipeline_timestamp}")
    print(f"📂 根目录: {root_dir}")
    
    client_env = os.environ.copy()
    


    for r in range(TOTAL_ROUNDS):
        print(f"\n\n{'#'*80}")
        print(f"🔥🔥🔥 进入第 {r} 轮迭代 (Round {r}) 🔥🔥🔥")
        print(f"{'#'*80}")

        curr_paths = get_round_paths(root_dir, pipeline_timestamp, r, tag=EXP_TAG)
        prev_paths = get_round_paths(root_dir, pipeline_timestamp, r-1, tag=EXP_TAG) if r > 0 else None

        # ==============================================================================
        # 🧠 [核心逻辑] 定义本轮 Cluster/Optimizer 的“输入源” (Input Source)
        # 这就是你要求的跨文件逻辑：Round 0 读当前，Round N 读上一轮的 After
        # ==============================================================================
        if r == 0:
            print(f"📌 [Round 0] 模式：输入源为本轮 Pre 生成的初始文件")
            input_corpus = curr_paths['corpus']
            input_stats  = curr_paths['stats']
            input_freq   = curr_paths['freq']
        else:
            print(f"📌 [Round {r}] 模式：输入源为 Round {r-1} Eval 生成的 After 文件")
            # 🔥 这里的接力就是你要求的核心：
            input_corpus = prev_paths['optimized_memory'] # 上一轮优化后的记忆
            input_stats  = prev_paths['stats_after']      # 上一轮 Eval 后的 stats
            input_freq   = prev_paths['freq_after']       # 上一轮 Eval 后的 freq
            
            # 安全检查：确保上一轮真的把接力棒递过来了
            for f_path, f_name in [(input_corpus, "Optimized Memory"), (input_stats, "Stats After"), (input_freq, "Freq After")]:
                if not os.path.exists(f_path):
                    print(f"❌ 致命错误：上一轮的 {f_name} 不存在！路径: {f_path}")
                    print("   可能上一轮 Eval 没跑完或者没生成 _after 文件。")
                    sys.exit(1)

        # --------------------------------------------------
        # Step 1: Pre-process (仅 Round 0 需要)
        # --------------------------------------------------
        if r == 0:
            pre_overrides = {
                # Pre 输出到当前轮的 stats/corpus/freq
                "paths.stats_file": curr_paths['stats'],
                "paths.freq_file": curr_paths['freq'], 
                "paths.corpus_file": curr_paths['corpus'],
                "paths.test_file": curr_paths['test'],
                "paths.result_dir": curr_paths['dir'], 
            }
            # 第一轮时的evallast：
            run_step("evallast.py", f"首先进行一个测试集测试，进行效果查看",overrides = pre_overrides, env=client_env)
            # 如果是第一次，可能没有 stats 文件，prepro.py 会生成它
            run_step("prepro.py", f"R{r}-1. 初始数据准备", pre_overrides, env=client_env)

        # --------------------------------------------------
        # Step 2: Clustering
        # --------------------------------------------------
        # Cluster 读取我们在上面定义好的 input_xxx
        cluster_overrides = {
            # 输出路径 (当前轮)
            "paths.cluster_output": curr_paths['cluster_output'],
            "paths.cluster_summary": curr_paths['cluster_summary'],
            "paths.cluster_vis": curr_paths['cluster_vis'],
            "paths.cluster_plot": curr_paths['cluster_plot'],
            
            # 🔥 输入路径 (动态源)
            "paths.corpus_file": input_corpus,  # 读谁的记忆库？
            "paths.stats_file": input_stats,    # 读谁的 Stats？
            "paths.freq_file": input_freq       # 读谁的 Freq？
        }
        run_step("clusterpro.py", f"R{r}-2. 聚类", cluster_overrides, env=client_env)

        # --------------------------------------------------
        # Step 3: Optimizer
        # --------------------------------------------------
        opt_overrides = {
            # 输入 (与 Cluster 一致)
            "paths.corpus_file": input_corpus,
            "paths.stats_file": input_stats,
            "paths.freq_file": input_freq,
            
            # Cluster 的结果
            "paths.cluster_output": curr_paths['cluster_output'],
            "paths.cluster_summary": curr_paths['cluster_summary'],
            
            # 🔥 输出 (当前轮的新记忆和重置Stats)
            "paths.optimized_memory": curr_paths['optimized_memory'],
            "paths.stats_optimized_file": curr_paths['stats_optimized'], 
        }
        run_step("optimizerultra.py", f"R{r}-3. 记忆优化", opt_overrides, env=client_env)

        # --------------------------------------------------
        # Step 4: Eval (生成 After 文件)
        # --------------------------------------------------
        
        eval_overrides = {
            # Eval 评测的是刚刚优化好的记忆
            "paths.corpus_file": curr_paths['optimized_memory'],
            
            # Eval 读取 Optimizer 重置后的 Stats (作为起点)
            "paths.stats_optimized_file": curr_paths['stats_optimized'],
            
            # 🔥 关键：Eval 跑完后，要把结果写到 _after 文件里，供下一轮 Cluster 读取！
            "paths.stats_after_file": curr_paths['stats_after'],
            "paths.freq_after_file": curr_paths['freq_after'],
            
            "paths.rag_cache_dir": curr_paths['rag_cache'],
            "parameters.is_first": False,

            "paths.result_dir": curr_paths['dir'], 
        }
        
        # 检查 Optimizer 是否成功产出
        if not os.path.exists(curr_paths['stats_optimized']):
            # 兜底逻辑：如果 Opt 没产出，就拷贝 input_stats 过来假装它是优化后的
            import shutil
            print(f"⚠️ 警告：Optimizer 未生成 Stats，沿用输入 Stats。")
            shutil.copy(input_stats, curr_paths['stats_optimized'])

        run_step("evalpro.py", f"R{r}-4. 效果评测 & 更新After状态", eval_overrides, env=client_env)
        if r < TOTAL_ROUNDS - 1:
            # print("跳过")
            run_step("evallast.py", f"R{r}-5. 测试集测试，效果查看", eval_overrides, env=client_env)
        else:
            run_step("evallast.py", f"R{r}-5. 测试集测试，效果查看", eval_overrides, env=client_env)

        print(f"\n 一轮测试执行完毕！")


    print(f"\n🎉🎉🎉 全流程执行完毕！")

if __name__ == "__main__":
    main()