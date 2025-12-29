import os
import json
from typing import Dict, List
# Hydra
import hydra
from omegaconf import DictConfig
from tools.optimize.callllm import init_llm, call_llm_batch
from tools.optimize.callexpert import init_expert_llm, call_expert,call_expert_batch
from tools.optimize.memoryload import load_clustered_memories, load_cluster_summary

# ==========================================
# 1. Prompt 构造函数 (TextGrad 双阶段)
# ==========================================

def generate_gradient_prompt(content: str, neg_queries: List[str]) -> str:
    """
    🔥 [Step 1: Backward Pass - Expert]
    请求专家模型进行“归因分析”，计算文本梯度。
    """
    neg_text = "\n".join([f"- {q}" for q in neg_queries[:5]])
    prompt = f"""
You are a Senior Knowledge Engineer diagnosing a RAG system memory.

[Target Memory]
{content}

[Failure Cases]
The system used this memory to answer the following queries but failed:
{neg_text}

[Task: Calculate Gradient]
Analyze WHY this memory failed.
- Is it missing specific formulas or conditions?
- Is it ambiguous?
- Is it a "Hubness" problem (irrelevant but high similarity)?

Provide a concise **Improvement Instruction** (The Gradient).
Start with "To fix this, you should..."
"""
    return prompt

def apply_gradient_prompt(content: str, gradient: str, good_examples: str, cfg: DictConfig) -> str:
    """
    🔥 [Step 2: Update Step - Student]
    请求 Qwen 根据专家的梯度重写记忆。
    """
    momentum_part = ""
    if good_examples:
        momentum_part = f"\n[Reference (Momentum)]\nHigh-quality neighbors:\n{good_examples}\n"

    # 尝试读取 config 里的模板，否则用默认
    template = cfg.optimizer.prompts.apply_gradient
    return template.format(content=content, gradient=gradient, momentum_part=momentum_part)

def summarize_experience_prompt(target_text: str, good_neighbors: List[str], cfg: DictConfig) -> str:
    """旧逻辑：模仿"""
    good_examples_text = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(good_neighbors))
    template = cfg.optimizer.prompts.expand_low_freq
    prompt = template.format(text=target_text, good_examples=good_examples_text)
    return prompt

def expand_low_freq_memory_prompt(text: str, good_examples: str, cfg: DictConfig) -> str:
    """旧逻辑：自省"""
    template = cfg.optimizer.prompts.expand_low_freq
    prompt = template.format(text=text, good_examples=good_examples)
    return prompt

# ==========================================
# 2. 筛选逻辑 (保持不变)
# ==========================================
def select_ids_from_stats(memory_stats: Dict[str, dict], cfg: DictConfig):
    scores = []
    for mid, stats in memory_stats.items():
        alpha = stats.get('alpha', 1.0)
        beta = stats.get('beta', 1.0)
        total = alpha + beta
        win_rate = alpha / total if total > 0 else 0.5
        scores.append((mid, win_rate, total))
    
    scores.sort(key=lambda x: (-x[1], -x[2]))
    
    top_k = cfg.optimizer.top_k_high
    bottom_k = cfg.optimizer.bottom_k_low
    
    high_ids = [x[0] for x in scores[:top_k]]
    # 筛选有错误记录的 ID
    bad_ids = [x[0] for x in scores if x[1] < 0.4 and x[2] > 2]
    
    if len(bad_ids) < 10:
         bad_ids = [x[0] for x in scores[-bottom_k:]]

    print(f"🔥 高分 Anchor: {len(high_ids)}")
    print(f"🥶 低分 Candidates: {len(bad_ids)}")
    return set(high_ids), set(bad_ids)

# ==========================================
# 3. 主程序
# ==========================================
@hydra.main(version_base=None, config_path="conf", config_name="config")
def optimize_memory(cfg: DictConfig):
    # 0. 初始化双模型
    init_llm(cfg)          # 学生 (Qwen)
    init_expert_llm(cfg)   # 专家 (Gemini/GPT)

    # 1. 路径
    cluster_file = cfg.paths.cluster_output
    summary_file = cfg.paths.cluster_summary
    stats_file = cfg.paths.stats_file
    output_file = cfg.paths.optimized_memory
    
    root_dir = cfg.paths.root
    corpus_name = cfg.experiment.get("corpus_dataset_name") or cfg.experiment.dataset_name
    corpus_tag = corpus_name.split('/')[-1]
    # 优化后的 Stats 保存路径
    stats_optimized_file = cfg.paths.stats_optimized_file

    # 2. 加载数据
    if not os.path.exists(stats_file):
        print(f"❌ 找不到状态文件: {stats_file}")
        return
    with open(stats_file, 'r', encoding='utf-8') as f:
        memory_stats = json.load(f)

    memories, id_order = load_clustered_memories(cluster_file)
    cluster_to_ids = load_cluster_summary(summary_file)
    if not memories: return

    # 3. 筛选
    high_ids, bad_ids = select_ids_from_stats(memory_stats, cfg)

    # =========================================================
    # 4. Pruning (高分去噪)
    # =========================================================
    print("\n========== 高分记忆清理阶段 (Pruning) ==========")
    to_delete_ids = set() 
    
    cluster_groups = {}
    for mid, rec in memories.items():
        cid = rec.get("cluster_id")
        if cid is not None:
            cid = int(cid)
            if cid not in cluster_groups: cluster_groups[cid] = []
            cluster_groups[cid].append(mid)
    
    pruned_count = 0
    for cid, members in cluster_groups.items():
        if len(members) < 2: continue
        
        member_stats_list = []
        for mid in members:
            stats = memory_stats.get(mid, {'alpha': 1.0, 'beta': 1.0})
            total = stats['alpha'] + stats['beta']
            win_rate = stats['alpha'] / total if total > 0 else 0.5
            member_stats_list.append({'id': mid, 'win_rate': win_rate, 'total': total})
            
        member_stats_list.sort(key=lambda x: (-x['win_rate'], -x['total']))
        best_mem = member_stats_list[0]
        
        if best_mem['win_rate'] > 0.7 and best_mem['total'] > 2:
            for mem in member_stats_list[1:]:
                is_trash = False
                if mem['win_rate'] < 0.4 and mem['total'] > 2: is_trash = True
                if best_mem['win_rate'] > 0.9 and mem['win_rate'] < 0.5: is_trash = True
                if is_trash:
                    to_delete_ids.add(mem['id'])
                    pruned_count += 1
    print(to_delete_ids)
    print(f"✨ Pruning 完成，删除: {pruned_count}")

# =========================================================
    # 5. TextGrad (专家归因 -> 学生修正) - Batch 优化版
    # =========================================================
    print("\n========== TextGrad 记忆修正阶段 (Expert Batch Guided) ==========")
    low_expand_ids = [mid for mid in bad_ids if mid in memories and mid not in to_delete_ids]
    print(f"🎯 待优化目标数量: {len(low_expand_ids)}")

    # 使用 config 中的 batch size
    batch_size = cfg.optimizer.llm_batch_size
    
    # 记录优化状态
    optimized_ids = set()

    # 🔥 核心修改：按 Chunk 处理，实现全链路 Batch
    for i in range(0, len(low_expand_ids), batch_size):
        chunk_ids = low_expand_ids[i : i + batch_size]
        
        # --- Step 5.1: 准备专家 Prompts (Gradient Calculation) ---
        grad_prompts = []
        grad_metadata = [] # 存 (mid, base_text, good_examples_str, neg_queries_len)
        
        # 这一步不需要调 LLM，只是查表构建 Prompt
        for mid in chunk_ids:
            rec = memories[mid]
            base_text = rec.get("contents", "")
            cluster_id = rec.get("cluster_id")
            stats = memory_stats.get(mid, {})
            neg_queries = stats.get('neg_queries', [])
            
            # 找优等生 (Momentum)
            good_neighbors_text = []
            if cluster_id is not None:
                cluster_id = int(cluster_id)
                members = cluster_to_ids.get(cluster_id, [])
                for m_id in members:
                    if str(m_id) == mid: continue
                    s = memory_stats.get(str(m_id), {})
                    s_total = s.get('alpha', 0) + s.get('beta', 0)
                    if s_total > 0 and (s.get('alpha', 0)/s_total) > 0.8:
                        good_neighbors_text.append(memories[str(m_id)].get("contents", ""))
                good_neighbors_text = good_neighbors_text[:3]
            good_examples_str = "\n".join([f"- {t}" for t in good_neighbors_text])
            
            # 只有有错误记录的才需要专家介入
            if len(neg_queries) > 0:
                prompt = generate_gradient_prompt(base_text, neg_queries)
                grad_prompts.append(prompt)
                grad_metadata.append({
                    "mid": mid, 
                    "need_expert": True,
                    "base_text": base_text,
                    "good_examples_str": good_examples_str,
                    "good_neighbors_text": good_neighbors_text, # 备用
                    "err_count": len(neg_queries)
                })
            else:
                # 不需要专家的，标记一下，后面直接进学生 Prompt 构造
                grad_metadata.append({
                    "mid": mid, 
                    "need_expert": False,
                    "base_text": base_text,
                    "good_neighbors_text": good_neighbors_text,
                    "good_examples_str": good_examples_str, # 保持对齐
                })

        # --- Step 5.2: 批量调用专家 (Expert Batch) ---
        gradients = []
        if grad_prompts:
            print(f" 🧠 [Expert-Batch] 正在分析 {len(grad_prompts)} 条梯度...")
            gradients = call_expert_batch(grad_prompts, cfg)
        
        # --- Step 5.3: 准备学生 Prompts (Update Step) ---
        student_prompts = []
        student_metadata = []
        
        grad_idx = 0 # 游标，用于从 gradients 列表里取结果
        
        for meta in grad_metadata:
            mid = meta['mid']
            opt_type = "unknown"
            prompt = ""
            
            if meta['need_expert']:
                # 获取刚才专家的输出
                gradient_text = gradients[grad_idx] if grad_idx < len(gradients) else None
                grad_idx += 1
                
                if gradient_text:
                    # 成功拿到梯度 -> TextGrad Update
                    prompt = apply_gradient_prompt(meta['base_text'], gradient_text, meta['good_examples_str'], cfg)
                    opt_type = f"textgrad_{meta['err_count']}_errors"
                else:
                    # 专家调用失败 -> 降级为 Imitation
                    if meta['good_neighbors_text']:
                        prompt = summarize_experience_prompt(meta['base_text'], meta['good_neighbors_text'], cfg)
                        opt_type = "neighbor_imitation"
                    else:
                        prompt = expand_low_freq_memory_prompt(meta['base_text'], "", cfg)
                        opt_type = "self_reflection"
            else:
                # 不需要专家 -> Imitation or Reflection
                if meta['good_neighbors_text']:
                    prompt = summarize_experience_prompt(meta['base_text'], meta['good_neighbors_text'], cfg)
                    opt_type = "neighbor_imitation"
                else:
                    prompt = expand_low_freq_memory_prompt(meta['base_text'], "", cfg)
                    opt_type = "self_reflection"
            
            student_prompts.append(prompt)
            student_metadata.append({"mid": mid, "opt_type": opt_type})

        # --- Step 5.4: 批量调用学生 (Student Batch) ---
        if student_prompts:
            print(f" 🚀 [Student-Batch] 正在优化 {len(student_prompts)} 条记忆...")
            outputs = call_expert_batch(student_prompts, cfg)
            
            # 回填结果
            for meta, output_text in zip(student_metadata, outputs):
                if output_text and len(output_text) > 10:
                    mid = meta['mid']
                    memories[mid]["contents"] = output_text
                    memories[mid]["opt_type"] = meta['opt_type']
                    optimized_ids.add(mid)

    # =========================================================
    # 6. 写出新记忆库
    # =========================================================
    print("\n========== 写出优化后的记忆库 ==========")
    kept_count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for mid in id_order:
            if mid not in memories: continue
            if mid in to_delete_ids: continue
            f.write(json.dumps(memories[mid], ensure_ascii=False) + "\n")
            kept_count += 1
    print(f"✅ 记忆库已保存: {output_file}")

    # =========================================================
    # 7. 状态同步 (Clean & Reset)
    # =========================================================
    print("\n========== 同步 BEMR 状态 (Stats Sync) ==========")
    
    # 1. 移除已删除的 ID
    for del_id in to_delete_ids:
        if del_id in memory_stats:
            del memory_stats[del_id]
            
    # 2. 重置已优化的 ID (Reset to Prior)
    for opt_id in optimized_ids:
        if opt_id in memory_stats:
            memory_stats[opt_id] = {
                'alpha': 1.0, 
                'beta': 1.0, 
                'pos_queries': [], 
                'neg_queries': [] # 清空梯度，因为已经修好了
            }
            
    print(f"   🗑️ 已从 Stats 中移除 {len(to_delete_ids)} 条")
    print(f"   🔄 已重置 {len(optimized_ids)} 条 TextGrad 优化项")
    
    try:
        with open(stats_optimized_file, 'w', encoding='utf-8') as f:
            json.dump(memory_stats, f, ensure_ascii=False, indent=2)
        print(f"✅ [BEMR] 状态已同步: {stats_optimized_file}")
    except Exception as e:
        print(f"❌ 保存 Stats 失败: {e}")

if __name__ == "__main__":
    optimize_memory()