import os
import json
import time
import torch
from huggingface_hub import snapshot_download
# Hydra & OmegaConf
import hydra
from omegaconf import DictConfig, OmegaConf
import random # [新增] GPQA 打乱选项需要
# FlashRAG
from flashrag.config import Config
from flashrag.pipeline import SequentialPipeline
from flashrag.utils import get_retriever, get_generator, Dataset
from flashrag.prompt import PromptTemplate

# 屏蔽 transformers 的冗余警告 和 httpx 的 INFO 日志 
import transformers
transformers.logging.set_verbosity_error()

# ==========================================
# 1. 一波引用
# ==========================================
from utils.prepare_data import prepare_data
from utils.build_index import build_index
from utils.generator.gemini import GeminiGenerator
from utils.generator.sglang import SGLangGenerator
from tools.evaluate import judge_math_item,evaluate_results

# ==========================================
# 2. 核心功能函数 (Hydra 适配)
# ==========================================
from tools.memoryscore import _load_memory_corpus,_calculate_scores,_print_stats_and_save,_visualize_results

def analyze_memory_usage(rag_results, cfg: DictConfig, corpus_file: str, vis_image_file: str):
    """
    记忆热度/效用统计与导出 (强化学习版) - 主入口
    逻辑：
    - 检索命中 & 题目做对: freq += 2 (奖励)
    - 检索命中 & 题目做错: freq -= 1 (惩罚)
    """
    freq_file = cfg.paths.freq_file
    print("\n🔍 [Analysis] 正在进行全量记忆效用评分 (RL Scoring)...")

    # 1. 加载数据
    all_memory_ids, id_to_content = _load_memory_corpus(corpus_file)

    # 2. 计算分数
    memory_scores, correct_count = _calculate_scores(rag_results, all_memory_ids, cfg)

    # 3. 打印统计并保存文件 (需要返回排序后的列表供可视化使用)
    sorted_memories = _print_stats_and_save(
        memory_scores, 
        id_to_content, 
        len(rag_results), 
        correct_count, 
        freq_file
    )

    # 4. 可视化或打印 Top 10
    _visualize_results(cfg, sorted_memories, vis_image_file)
# ==========================================
# 4. 主程序 (Hydra Managed)
# ==========================================

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    # 0. 基础设置与路径构造
    print("Visible GPU count:", torch.cuda.device_count())
    
    root_dir = cfg.paths.root
    
    # =================================================================
    # 🔥 [修改点 1] 分别提取“记忆库标签”和“测试集标签”
    # =================================================================
    # 优先读取 xxx_dataset_name，如果 yaml 里没写，回退到 dataset_name
    
    # 1. 记忆库 Tag (用于命名 corpus 和 index)
    corpus_name = cfg.experiment.get("corpus_dataset_name") or cfg.experiment.dataset_name
    corpus_tag = corpus_name.split('/')[-1] 
    
    # 2. 测试集 Tag (用于命名 test_data 和 log)
    test_name = cfg.experiment.get("test_dataset_name") or cfg.experiment.dataset_name
    test_tag = test_name.split('/')[-1]

    print(f"🏷️  Corpus Tag: {corpus_tag} | Test Tag: {test_tag}")

    # =================================================================
    # 🔥 [修改点 2] 文件名分离
    # =================================================================
    
    # 记忆库文件 & 索引目录 -> 跟随 corpus_tag (比如 MATH)
    corpus_file = os.path.join(root_dir, f"{corpus_tag}_corpus.jsonl")
    index_dir = os.path.join(root_dir, f"{corpus_tag}_bm25_index")
    
    # 测试集数据文件 -> 跟随 test_tag (比如 hmmt)
    # 这样你就不会把 MATH 的测试集覆盖掉了
    test_file = os.path.join(root_dir, f"{test_tag}_test_data.jsonl")
    
    # 结果日志 -> 最好同时体现 "用什么库测什么题"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 格式: HMMT_on_MATH_sglang_rag_2025...
    result_log_file = os.path.join(root_dir, f"{test_tag}_on_{corpus_tag}_{cfg.model.source}_{cfg.experiment.mode}_{timestamp}.txt")
    
    # 可视化图片 -> 跟随日志名
    vis_image_file = os.path.join(root_dir, f"{test_tag}_on_{corpus_tag}_dist_{timestamp}.png")

    if os.path.exists(result_log_file): os.remove(result_log_file)
    print(f"📝 结果将保存至: {result_log_file}")
    print(f"🛠️ 模式: {cfg.experiment.mode} | 源: {cfg.model.source}")
    print(f"📚 Memory: {corpus_name} | 🎯 Test: {test_name}")

    # 1. 数据准备
    if not prepare_data(cfg, corpus_file, test_file): return
    
    # 2. 索引构建 (如果是 rag 或 all 模式)
    if cfg.experiment.mode in ['rag', 'all']:
        build_index(corpus_file, index_dir)
    
    # 3. 初始化 Generator
    generator = None
    config = None # FlashRAG config
    
    model_source = cfg.model.source
    
    if model_source == "gemini":
        print(f"🤖 [Init] 初始化 Gemini: {cfg.model.gemini_name}...")
        api_key = os.environ.get("GEMINI_API_KEY") 
        generator = GeminiGenerator(cfg.model.gemini_name, api_key)
        
        # 构造 FlashRAG 配置字典
        gemini_config_dict = {
            "data_dir": root_dir,
            "save_dir": cfg.paths.rag_cache_dir,
            "device": "cpu",
            "retrieval_method": cfg.experiment.retrieval_method,
            "corpus_path": corpus_file,
            "index_path": index_dir,
            "retriever_model_path": index_dir,
            "generator_model": "huggingface", # 占位
            "generator_model_path": "gpt2",   # 占位
        }
        config = Config(config_dict=gemini_config_dict)

    elif model_source == "sglang":
        print(f"🚀 [Init] 初始化 SGLang Client...")
        
        # 1. 从 config 读取
        sglang_base_url = cfg.model.get("sglang_api_url", "http://127.0.0.1:30000/v1")
        # ⚠️ 确保这里读到的是 "Qwen/Qwen3-4B-Instruct-2507"
        sglang_model_name = cfg.model.get("sglang_model_name", "Qwen/Qwen3-4B-Instruct-2507")
        
        # 2. 构造 FlashRAG 配置字典
        sglang_config_dict = {
            "data_dir": root_dir,
            "save_dir": cfg.paths.rag_cache_dir,
            "corpus_path": corpus_file,
            "index_path": index_dir,
            "retriever_model_path": index_dir,
            "retrieval_method": cfg.experiment.retrieval_method,
            
            # --- 关键修改 ---
            "device": "cpu",
            "gpu_num": 0,
            "generator_model": "openai",               
            "generator_model_path": sglang_model_name, 
            "generation_method": "openai", 
            "batch_size": cfg.model.batch_size,
            "max_input_len": cfg.model.max_input_len,
            "max_new_tokens": cfg.model.max_new_tokens,
        }
        
        config = Config(config_dict=sglang_config_dict)

        # 4. 初始化 Generator
        generator = SGLangGenerator(
            base_url=sglang_base_url,
            model_name=sglang_model_name,
            max_new_tokens=cfg.model.max_new_tokens,
            batch_size=cfg.model.batch_size,
            temperature=0.7, # 如果需要，这个也可以提到 yaml 里
        )
        print(f"✅ SGLang Generator ({sglang_model_name}) 已连接至 {sglang_base_url}")

    elif model_source == "huggingface":
        hf_name = cfg.model.hf_name
        print(f"📥 [Init] 检查/下载 HF 模型: {hf_name}...")
        try:
            model_path = snapshot_download(repo_id=hf_name)
        except:
            print("❌ 模型下载失败")
            return

        hf_config_dict = {
            "data_dir": root_dir,
            "save_dir": cfg.paths.rag_cache_dir,
            "device": cfg.model.device,
            "gpu_num": torch.cuda.device_count(),
            "generator_model": "huggingface",
            "generator_model_path": model_path,
            "generation_method": "huggingface",
            "batch_size": cfg.model.batch_size,
            "max_input_len": cfg.model.max_input_len,
            "max_new_tokens": cfg.model.max_new_tokens,
        }
        print("🚀 加载 HF 生成器...")
        config = Config(config_dict=hf_config_dict)
        generator = get_generator(config)
        
        # 🔥 Tokenizer 修正 (保持你原有的 padding 修复)
        if hasattr(generator, 'tokenizer'):
            generator.tokenizer.padding_side = 'left' 
            if generator.tokenizer.pad_token is None:
                generator.tokenizer.pad_token = generator.tokenizer.eos_token
                generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id
            generator.tokenizer.model_max_length = cfg.model.max_input_len
        
        if hasattr(generator, 'model'):
            if hasattr(generator.model.config, 'pad_token_id') and generator.model.config.pad_token_id is None:
                generator.model.config.pad_token_id = generator.tokenizer.pad_token_id
        print(f"✅ Tokenizer 修正完成")
    
    else:
        print(f"❌ 未支持的 MODEL_SOURCE: {model_source}")
        return

    # 读取测试数据
    with open(test_file, "r") as f:
        test_dataset_raw = [json.loads(line) for line in f]

    acc_baseline = 0
    acc_rag = 0

    # 格式化 Prompt 辅助函数
    def format_base_prompt(system_text, user_text):
        if model_source == "gemini":
            return f"{system_text}\n\n{user_text}" if system_text else user_text
        prompt = ""
        if system_text: prompt += f"{system_text}\n\n"
        prompt += f"### Question:\n{user_text}\n\n### Answer:\nLet's think step by step."
        return prompt

    # --- Task A: Baseline ---
    if cfg.experiment.mode in ['baseline', 'all']:
        print("\n⚔️ [Task A] 正在运行 Baseline ...")
        
        baseline_inputs = []
        for item in test_dataset_raw:
            sys_msg = cfg.experiment.prompts.sys_msg
            formatted_prompt = format_base_prompt(sys_msg, item['question'])
            baseline_inputs.append(formatted_prompt)

        baseline_preds = generator.generate(baseline_inputs)
        
        baseline_results = []
        for item, pred in zip(test_dataset_raw, baseline_preds):
            baseline_results.append({
                "question": item['question'],
                "golden_answers": item['golden_answers'],
                "pred": pred
            })
        acc_baseline = evaluate_results(baseline_results, "Baseline (No RAG)", result_log_file)

    # --- Task B: FlashRAG ---
    if cfg.experiment.mode in ['rag', 'all']:
        print("\n⚔️ [Task B] 正在运行 FlashRAG (Few-shot Retrieval)...")
        
        # 准备 RAG 配置
        rag_config_dict = OmegaConf.to_container(cfg, resolve=True) # 仅仅是为了获取一些基础类型
        # 将 FlashRAG 需要的特定字段覆盖进去
        rag_update = {
            "data_dir": root_dir,
            "save_dir": cfg.paths.rag_cache_dir,
            "retrieval_method": cfg.experiment.retrieval_method,
            "corpus_path": corpus_file,
            "index_path": index_dir,
            "retriever_model_path": index_dir,
            "topk": cfg.experiment.retrieval_topk,
            # Generator 配置继承之前的
            "device": cfg.model.device,
            "generator_model_path": config['generator_model_path'] if 'generator_model_path' in config else "gpt2"
        }
        
        # 重新实例化 Config 以确保 Retriever 能读到正确参数
        rag_config = Config(config_dict=rag_update)
        retriever = get_retriever(rag_config)
        
        rag_system_part = cfg.experiment.prompts.rag_system
        
        prompt_tpl = PromptTemplate(rag_config, system_prompt=rag_system_part, user_prompt="")
        pipeline = SequentialPipeline(rag_config, prompt_template=prompt_tpl, retriever=retriever, generator=generator)
        dataset_obj = Dataset(rag_config, test_file)
        
        rag_results = pipeline.run(dataset_obj)
        
        acc_rag = evaluate_results(rag_results, f"FlashRAG ({corpus_tag} Memory)", result_log_file)
        
        # 统计记忆热度 (传入 cfg)
        analyze_memory_usage(rag_results, cfg, corpus_file, vis_image_file)

    # --- Summary ---
    if cfg.experiment.mode == 'all':
        summary = (
            f"\n{'='*20} 最终对比结果 {'='*20}\n"
            f"📊 数据集: {cfg.experiment.test_dataset_name}\n"
            f"🤖 模型: {model_source}\n"
            f"📉 Baseline: {acc_baseline:.2f}%\n"
            f"📈 FlashRAG: {acc_rag:.2f}%\n"
            f"🚀 提升: {acc_rag - acc_baseline:+.2f}%\n"
            f"{'='*50}\n"
        )
        print(summary)
        with open(result_log_file, "a", encoding="utf-8") as f:
            f.write(summary)

if __name__ == "__main__":
    main()