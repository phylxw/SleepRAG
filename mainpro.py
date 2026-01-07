import subprocess
import sys
import os
import time
from datetime import datetime
import hydra
from omegaconf import DictConfig
from utils.logger import setup_logging,Logger
import logging
import shutil

# 🤫 把 httpx 和 httpcore 的日志级别调高到 WARNING
# 这样只有出错才会打印，正常的 200 OK 就不显示了
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


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
    
    # 🔥 [新增] 获取断点续训路径 (例如: "/root/.../round_9")
    RESUME_PATH = cfg.parameters.get("resume_path", None) 
    
    # 路径修正
    root_dir = cfg.paths.root if "paths" in cfg and "root" in cfg.paths else os.getcwd()
    root_dir = os.path.abspath(root_dir)

    # 1. 初始化
    pipeline_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{EXP_TAG}_Loop"
    setup_logging(root_dir, pipeline_timestamp) 
    print(f"\n🎬 [Pipeline Start] 多轮迭代任务 | ID: {pipeline_timestamp}")
    
    if RESUME_PATH:
        print(f"🔄 [Resume Mode] 检测到续训路径: {RESUME_PATH}")
        print(f"    将基于该目录的产出作为 Round 0 的起点")
    
    print(f"📂 根目录: {root_dir}")
    
    client_env = os.environ.copy()

    for r in range(TOTAL_ROUNDS):
        print(f"\n\n{'#'*80}")
        print(f"🔥🔥🔥 进入第 {r} 轮迭代 (Round {r}) 🔥🔥🔥")
        print(f"{'#'*80}")

        curr_paths = get_round_paths(root_dir, pipeline_timestamp, r, tag=EXP_TAG)
        prev_paths = get_round_paths(root_dir, pipeline_timestamp, r-1, tag=EXP_TAG) if r > 0 else None
        
        # 标记是否跳过 Prepro (只有 R0 且 Resume 模式下跳过)
        skip_prepro = False

        # ==============================================================================
        # 🧠 [核心逻辑] 定义输入源 (Input Source)
        # ==============================================================================
        if r == 0:
            if RESUME_PATH and os.path.exists(RESUME_PATH):
                # === 💡 模式 A: 断点续训 (Resume) ===
                print(f"📌 [Round 0 - Resume] 模式：接力上一轮结果 -> {RESUME_PATH}")
                
                # 映射规则：把上一轮的“终态”当作这一轮的“初态”
                # 注意：这里假设 Resume 文件夹里的文件名 Tag 和当前 Tag 一致。
                # 如果不一致，你可能需要手动改一下 Resume 文件夹里的文件名，或者代码里做模糊匹配。
                
                # 1. 记忆库：上一轮优化后的结果
                input_corpus = os.path.join(RESUME_PATH, f"{EXP_TAG}_optimized_memory_topk.jsonl")
                if not os.path.exists(input_corpus):
                    # 容错：如果上一轮没跑完优化，试试读 corpus
                    print("⚠️ 没找到 optimized_memory，尝试读取 corpus...")
                    input_corpus = os.path.join(RESUME_PATH, f"{EXP_TAG}_corpus.jsonl")

                # 2. 状态与频次：上一轮 Eval 后的最终状态
                input_stats = os.path.join(RESUME_PATH, f"{EXP_TAG}_memory_after_stats.json")
                input_freq  = os.path.join(RESUME_PATH, f"{EXP_TAG}_memory_after_freq.jsonl")
                
                skip_prepro = True # 既然是接力，就不要重新初始化了

                # 🚀 [动作] 把这些“先验知识”拷贝到当前 Round 0 的目录下
                # 这样做的好处是：Round 0 的文件夹里会有一份完整的起点数据，方便后续追溯
                print(f"📦 正在迁移先验知识到当前目录...")
                if os.path.exists(input_stats): shutil.copy(input_stats, curr_paths['stats'])
                if os.path.exists(input_freq):  shutil.copy(input_freq, curr_paths['freq'])
                if os.path.exists(input_corpus): shutil.copy(input_corpus, curr_paths['corpus'])
                
                # 修正：虽然 input 指向了 ResumePath，但为了 Cluster/Opt 能读到“当前轮”的标准路径，
                # 我们这里可以偷懒，直接把 input 指向刚刚拷贝过来的 curr_paths
                input_corpus = curr_paths['corpus']
                input_stats  = curr_paths['stats']
                input_freq   = curr_paths['freq']

            else:
                # === 💡 模式 B: 冷启动 (Fresh Start) ===
                print(f"📌 [Round 0 - Fresh] 模式：全流程初始化")
                input_corpus = curr_paths['corpus']
                input_stats  = curr_paths['stats']
                input_freq   = curr_paths['freq']
                
        else:
            # === 💡 模式 C: 正常循环 (Loop) ===
            print(f"📌 [Round {r}] 模式：输入源为 Round {r-1} Eval 生成的 After 文件")
            input_corpus = prev_paths['optimized_memory']
            input_stats  = prev_paths['stats_after']
            input_freq   = prev_paths['freq_after']

        # 安全检查
        # 🔥 [Fix] 只有在 "非冷启动" 的情况下才检查文件是否存在
        # 冷启动时(Fresh)，文件还没生成呢，要等后面的 prepro.py 来生成
        is_fresh_start = (r == 0 and not RESUME_PATH)

        if not is_fresh_start:
            for f_path, f_name in [(input_corpus, "Corpus"), (input_stats, "Stats"), (input_freq, "Freq")]:
                if not os.path.exists(f_path):
                    print(f"❌ 致命错误：输入文件 {f_name} 不存在！路径: {f_path}")
                    if r == 0 and RESUME_PATH:
                        print("💡 提示：请检查 resume_path 下的文件名是否包含 tag 前缀。")
                    sys.exit(1)
        else:
            print("🌱 [Fresh Start] 初始文件将在 Step 1 (Prepro) 中生成，跳过存在性检查。")

        # --------------------------------------------------
        # Step 1: Pre-process
        # --------------------------------------------------
        if r == 0:
            # 1. 定义基础参数 (无论冷启动还是续训都需要)
            pre_overrides = {
                "paths.stats_file": curr_paths['stats'],
                "paths.freq_file": curr_paths['freq'], 
                "paths.corpus_file": curr_paths['corpus'],
                "paths.test_file": curr_paths['test'],
                "paths.result_dir": curr_paths['dir'], 
            }
            
            # 2. 根据模式决定是否“魔改”读取路径
            if skip_prepro:
                # === 🔄 Resume 模式 ===
                print("⏩ [Resume] 跳过数据初始化 (Prepro)...")
                
                # 🔥 [关键修复] 只有在 Resume 时，才强制指定 optimized_memory 为当前拷贝过来的 corpus
                # 这样 evallast 就会测试我们从上一轮继承过来的记忆
                eval_overrides = {
                    "paths.corpus_file": curr_paths['corpus'],
                    "paths.stats_optimized_file": curr_paths['stats'],
                    "paths.stats_after_file": curr_paths['stats_after'],
                    "paths.freq_after_file": curr_paths['freq_after'],
                    "paths.rag_cache_dir": curr_paths['rag_cache'],
                    "parameters.is_first": False,
                    "paths.result_dir": curr_paths['dir'], 
                }
                
                # run_step("evallast.py", f"R{r}-0. 接力起点(Resume)效果测试", overrides=eval_overrides, env=client_env)
                
            else:
                # === 🌱 Fresh 模式 ===
                # 在这里，我们不覆盖 "paths.optimized_memory"。
                # evallast.py 会使用 config.yaml 里默认配置的路径（通常是空的或者指向原始数据集），
                # 或者在代码里有兜底逻辑（如果找不到 opt 就测 raw）。
                # 这样就避免了指向一个还不存在的文件。
                
                run_step("evallast.py", f"R{r}-0. 初始Baseline测试", overrides=pre_overrides, env=client_env)
                run_step("prepro.py", f"R{r}-1. 初始数据准备", pre_overrides, env=client_env)

        # --------------------------------------------------
        # Step 2: Clustering
        # --------------------------------------------------
        cluster_overrides = {
            "paths.cluster_output": curr_paths['cluster_output'],
            "paths.cluster_summary": curr_paths['cluster_summary'],
            "paths.cluster_vis": curr_paths['cluster_vis'],
            "paths.cluster_plot": curr_paths['cluster_plot'],
            "paths.corpus_file": input_corpus,
            "paths.stats_file": input_stats,
            "paths.freq_file": input_freq
        }
        run_step("clusterpro.py", f"R{r}-2. 聚类", cluster_overrides, env=client_env)

        # --------------------------------------------------
        # Step 3: Optimizer
        # --------------------------------------------------
        opt_overrides = {
            "paths.corpus_file": input_corpus,
            "paths.stats_file": input_stats,
            "paths.freq_file": input_freq,
            "paths.cluster_output": curr_paths['cluster_output'],
            "paths.cluster_summary": curr_paths['cluster_summary'],
            "paths.optimized_memory": curr_paths['optimized_memory'],
            "paths.stats_optimized_file": curr_paths['stats_optimized'], 
        }
        # run_step("optimizerXtreme.py", f"R{r}-3. 记忆优化", opt_overrides, env=client_env)
        run_step("optimizerY.py", f"R{r}-3. 记忆优化", opt_overrides, env=client_env)
        # --------------------------------------------------
        # Step 4: Eval
        # --------------------------------------------------
        eval_overrides = {
            "paths.corpus_file": curr_paths['optimized_memory'],
            "paths.stats_optimized_file": curr_paths['stats_optimized'],
            "paths.stats_after_file": curr_paths['stats_after'],
            "paths.freq_after_file": curr_paths['freq_after'],
            "paths.rag_cache_dir": curr_paths['rag_cache'],
            "parameters.is_first": False,
            "paths.result_dir": curr_paths['dir'], 
        }

        # 兜底检查 Stats
        if not os.path.exists(curr_paths['stats_optimized']):
            print(f"⚠️ 警告：Optimizer 未生成 Stats，沿用输入 Stats。")
            shutil.copy(input_stats, curr_paths['stats_optimized'])

        run_step("evalpro.py", f"R{r}-4. 效果评测 & 更新After状态", eval_overrides, env=client_env)
        
        # 每轮最后跑一次测试集
        run_step("evallast.py", f"R{r}-5. 测试集验证", eval_overrides, env=client_env)

        print(f"\n✅ 第 {r} 轮执行完毕！")

    print(f"\n🎉🎉🎉 全流程执行完毕！")

if __name__ == "__main__":
    main()