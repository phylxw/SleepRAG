import os
import re
import uuid
import logging
from typing import List, Dict, Set, Any, Optional
from dataclasses import dataclass, field

# 假设工具库路径不变
from tools.optimize.callllm import call_llm_batch
from tools.optimize.callexpert import call_expert_batch
from utils.opt.toolfunction import _extract_single_memory, _basic_guard

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ICML_WarRoom")

@dataclass
class OptimizationTask:
    """追踪单个记忆的优化状态"""
    mid: str
    original_content: str
    stats: Dict
    # 诊断阶段
    diagnosis_prompt: str = ""
    expert_action: str = "WAITING" # REFINE, REPLACE, EXPAND
    expert_advice: str = ""
    # 执行阶段
    student_prompt: str = ""
    generated_content: str = ""
    # 评估阶段
    judge_verdict: str = "PENDING"
    judge_feedback: str = ""
    retry_count: int = 0
    # 结果
    final_accepted_content: Optional[str] = None
    # 🔥 [新增] 用于 EXPAND 逻辑
    is_new_node: bool = False 
    parent_id: Optional[str] = None

class TextGradOptimizer:
    def __init__(self, cfg, memories, memory_stats, log_path):
        self.cfg = cfg
        self.memories = memories
        self.memory_stats = memory_stats
        self.log_path = log_path
        self.batch_size = cfg.optimizer.llm_batch_size
        self.max_retries = cfg.parameters.get("max_retries", 2)
        
        # 预编译正则
        # 兼容两种格式：既能匹配 \box{EXPAND} 也能匹配 Action: EXPAND
        self.action_re = re.compile(r'(?:\\box\{|Action:\s*)(REFINE|EXPAND|REPLACE|CREATE)', re.IGNORECASE)
        
        # 兼容两种格式：既能匹配 \advice{...} 也能匹配 Advice: ...
        # 注意：Advice: 后面直到文本结束都算建议
        self.advice_re = re.compile(r'(?:\\advice\{|Advice:\s*)(.*?)(?:\}|(?=$))', re.DOTALL | re.IGNORECASE)
        self.verdict_re = re.compile(r"Verdict:\s*(PASS|FAIL)", re.IGNORECASE)

        # 清空日志文件头
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*20} New Optimization Session {'='*20}\n")

    def log(self, msg):
        """双写日志"""
        print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    def run(self, target_ids: List[str], to_delete_ids: Set[str]) -> Set[str]:
        """主入口"""
        # 1. 过滤有效ID
        valid_ids = [mid for mid in target_ids if mid in self.memories and mid not in to_delete_ids]
        self.log(f"🎯 待优化 {len(valid_ids)} 条记忆")
        
        optimized_ids = set()

        # 2. Batch Loop
        for i in range(0, len(valid_ids), self.batch_size):
            chunk_ids = valid_ids[i : i + self.batch_size]
            self.log(f"\n🚀 Processing Batch {i//self.batch_size + 1} ({len(chunk_ids)} items)")
            
            # 初始化任务对象
            tasks = self._init_tasks(chunk_ids)
            
            # Phase 1: Expert Diagnosis (诊断)
            # 这一步会填充 tasks 里的 expert_action 和 expert_advice
            self._batch_diagnose(tasks)
            
            # Phase 2: Student Execution (执行)
            # 🔥 [核心修改] 这里会处理 EXPAND，如果分裂，会返回新产生的 tasks
            new_tasks = self._batch_execute_and_expand(tasks)
            
            # 将分裂出来的新任务加入当前待评估列表
            if new_tasks:
                self.log(f"✨ [EXPAND Triggered] Added {len(new_tasks)} new split-nodes to current batch.")
                tasks.extend(new_tasks)

            # Phase 3: Judge Evaluation & Retry Loop (评估与修正)
            # 这一步会填充 final_accepted_content
            self._batch_evaluate_loop(tasks)
            
            # Phase 4: Commit (提交)
            # 这一步会将 final_accepted_content 写回 self.memories
            batch_success_ids = self._commit_changes(tasks)
            optimized_ids.update(batch_success_ids)
            
        return optimized_ids

    def _init_tasks(self, mids) -> List[OptimizationTask]:
        tasks = []
        for mid in mids:
            rec = self.memories[mid]
            tasks.append(OptimizationTask(
                mid=mid,
                original_content=rec.get("contents", ""),
                stats=self.memory_stats.get(mid, {})
            ))
        return tasks

    # --------------------------------------------------------------------------
    # Phase 1: Diagnosis (Expert)
    # --------------------------------------------------------------------------
    def _batch_diagnose(self, tasks: List[OptimizationTask]):
        prompts = []
        for task in tasks:
            neg_queries = task.stats.get('neg_queries', [])
            
            if not neg_queries:
                # 无错题模式 -> 润色
                prompt = self.cfg.optimizer.prompts.low_grad_polish.format(
                    content=task.original_content
                )
            else:
                # 错题模式 -> 专家诊断
                top_k_negs = "\n".join([f"- {q}" for q in neg_queries[:3]])
                prompt = self.cfg.optimizer.prompts.low_grad_expert.format(
                    content=task.original_content,
                    neg_queries=top_k_negs
                )
            
            task.diagnosis_prompt = prompt
            prompts.append(prompt)

        self.log(f"🧠 [Expert] Diagnosing {len(prompts)} memories...")
        outputs = call_expert_batch(prompts, self.cfg)

        for task, out in zip(tasks, outputs):
            if not out: continue
            # 解析 Action
            m_act = self.action_re.search(out)
            task.expert_action = m_act.group(1) if m_act else "REFINE" 
            
            # 解析 Advice/Gradient
            m_adv = self.advice_re.search(out)
            gradient = m_adv.group(1).strip() if m_adv else out.strip()
            task.expert_advice = gradient
            
            # 🔥 [日志增强] 打印具体梯度，就像原来那样
            preview = gradient[:60] + "..." if len(gradient) > 60 else gradient
            self.log(f"  -> ID:{task.mid[:6]} | Action: {task.expert_action}")
            self.log(f"     Gradient: {preview}")

    # --------------------------------------------------------------------------
    # Phase 2: Execution (Student) - [含 EXPAND 逻辑]
    # --------------------------------------------------------------------------
    def _batch_execute_and_expand(self, tasks: List[OptimizationTask]) -> List[OptimizationTask]:
        """
        根据专家建议，生成 Student Prompt。
        如果是 EXPAND，会生成两个 Prompt：
          1. 修改当前任务 (Refine)
          2. 创建新任务 (Create New) -> 返回这个新任务对象列表
        """
        prompts = []
        active_tasks = [] # 记录哪些任务发起了请求，用于回填 output
        new_spawned_tasks = [] # 存储 EXPAND 产生的新任务

        for task in tasks:
            if task.expert_action == "WAITING": continue
            
            neg_text = "\n".join(task.stats.get('neg_queries', [])[:3])
            gradient = task.expert_advice

            # --- 分发逻辑 ---
            if task.expert_action == "EXPAND":
                # 🔥 [复活 EXPAND 逻辑]
                # 1. 任务A：优化旧记忆 (Refine Old)
                p_old = self.cfg.optimizer.prompts.appgrad_low_refine.format(
                    content=task.original_content, 
                    gradient=f"Keep the general definition, but distinguish from new concept. Advice: {gradient}"
                )
                task.student_prompt = p_old
                prompts.append(p_old)
                active_tasks.append(task)

                # 2. 任务B：创建新记忆 (Create New)
                # 生成新 UUID
                new_mid = str(uuid.uuid4())
                
                # 初始化新任务对象
                new_task = OptimizationTask(
                    mid=new_mid,
                    original_content="", # 新记忆初始为空
                    stats={"neg_queries": task.stats.get('neg_queries', [])}, # 继承错题以便通过测试
                    expert_action="CREATE", # 标记为创建动作
                    is_new_node=True,
                    parent_id=task.mid
                )
                
                # 构建 Prompt (类似于 REPLACE，利用错题和梯度从头写)
                p_new = self.cfg.optimizer.prompts.appgrad_low_replace.format(
                    neg_queries=neg_text, 
                    gradient=f"Create a NEW memory specific to these queries. Advice: {gradient}"
                )
                new_task.student_prompt = p_new
                
                # 加入队列
                prompts.append(p_new)
                active_tasks.append(new_task) # 新任务也作为 active_task 接收 LLM 输出
                new_spawned_tasks.append(new_task)

            elif task.expert_action == "REPLACE":
                p = self.cfg.optimizer.prompts.appgrad_low_replace.format(neg_queries=neg_text, gradient=gradient)
                task.student_prompt = p
                prompts.append(p)
                active_tasks.append(task)

            else: # REFINE 
                p = self.cfg.optimizer.prompts.appgrad_low_refine.format(content=task.original_content, gradient=gradient)
                task.student_prompt = p
                prompts.append(p)
                active_tasks.append(task)

        if not prompts: return []

        self.log(f"✍️ [Student] Drafting updates for {len(prompts)} tasks (incl. expansions)...")
        
        # 批量调用 Student (建议用 call_llm_batch)
        outputs = call_expert_batch(prompts, self.cfg) 
        
        for t, out in zip(active_tasks, outputs):
            # 提取内容
            clean_content = _extract_single_memory(out)
            t.generated_content = clean_content if clean_content else out
            
            # 简单的日志
            if t.is_new_node:
                self.log(f"     [NEW NODE] Generated content for {t.mid[:6]} (Parent: {t.parent_id[:6]})")

        return new_spawned_tasks

    # --------------------------------------------------------------------------
    # Phase 3: Evaluation Loop (The Generalization Guard)
    # --------------------------------------------------------------------------
    def _batch_evaluate_loop(self, tasks: List[OptimizationTask]):
        """包含 Retry 的评估循环"""
        
        for retry_idx in range(self.max_retries + 1):
            # 1. 筛选需要评估的任务 (必须有生成内容，且还没PASS)
            pending_tasks = [t for t in tasks if t.judge_verdict != "PASS" and t.generated_content]
            if not pending_tasks:
                break
                
            self.log(f"⚖️ [Judge] Round {retry_idx}: Evaluating {len(pending_tasks)} candidates...")
            
            # 2. 构建 Judge Prompts
            judge_prompts = []
            for t in pending_tasks:
                neg_q = "\n".join(t.stats.get('neg_queries', [])[:3])
                # [泛化性检查]
                p = self.cfg.optimizer.prompts.expert_judge.format(failed = neg_q, old = t.original_content, new = t.generated_content)
                judge_prompts.append(p)
            
            # 3. 调用 Judge
            judge_outs = call_expert_batch(judge_prompts, self.cfg)
            
            # 4. 处理结果 & 准备 Retry
            retry_prompts = []
            retry_tasks = []
            
            for t, out in zip(pending_tasks, judge_outs):
                verdict_match = self.verdict_re.search(out)
                verdict = verdict_match.group(1).upper() if verdict_match else "FAIL"
                
                feedback = out.split("Feedback:")[-1].strip() if "Feedback:" in out else out[-100:]
                
                t.judge_verdict = verdict
                t.judge_feedback = feedback
                
                if verdict == "PASS":
                    t.final_accepted_content = t.generated_content
                    self.log(f"  ✅ [PASS] ID:{t.mid[:6]}")
                else:
                    self.log(f"  ❌ [FAIL] ID:{t.mid[:6]} | Feedback: {feedback[:50]}...")
                    if retry_idx < self.max_retries:
                        # 构建 Retry Prompt
                        new_prompt = self.cfg.optimizer.prompts.retry_prompt.format(ori = t.student_prompt, failed = neg_q, bad = t.generated_content,feedback = feedback)
                        retry_prompts.append(new_prompt)
                        retry_tasks.append(t)

            # 5. 执行 Retry 生成
            if retry_prompts:
                self.log(f"🔄 [Retry] Regenerating {len(retry_prompts)} items...")
                retry_outs = call_expert_batch(retry_prompts, self.cfg)
                for t, out in zip(retry_tasks, retry_outs):
                    t.generated_content = _extract_single_memory(out) or out
                    t.retry_count += 1
            else:
                break

    # --------------------------------------------------------------------------
    # Phase 4: Commit
    # --------------------------------------------------------------------------
    def _commit_changes(self, tasks: List[OptimizationTask]) -> Set[str]:
        success_ids = set()
        for t in tasks:
            if t.final_accepted_content:
                # 写入 Memory Storage
                if t.is_new_node:
                    # 🔥 [处理 EXPAND 新节点]
                    self.memories[t.mid] = {
                        "id": t.mid,
                        "contents": t.final_accepted_content,
                        "cluster_id": -1, # 等待重新聚类
                        "opt_type": "textgrad_expand",
                        "parent_id": t.parent_id
                    }
                    # 初始化 Stats
                    self.memory_stats[t.mid] = {
                        "alpha": 0.5, 
                        "beta": 0.5, 
                        "neg_queries": [], 
                        "pos_queries": []
                    }
                    self.log(f"✨ [EXPAND] Created New Node: {t.mid[:8]}")
                else:
                    # [处理 REFINE/REPLACE 旧节点]
                    self.memories[t.mid]["contents"] = t.final_accepted_content
                    self.memories[t.mid]["cluster_id"] = -1 
                    self.memories[t.mid]["opt_type"] = "textgrad_v2"
                    # 清空错题本
                    if t.mid in self.memory_stats:
                        self.memory_stats[t.mid]['neg_queries'] = []
                    self.log(f"💾 [UPDATE] Updated Node: {t.mid[:8]}")
                
                success_ids.add(t.mid)
        return success_ids

# ------------------------------------------------------------------------------
# 外部调用接口
# ------------------------------------------------------------------------------
def textgrad_opt(cfg, memories, memory_stats, log_file_path, cluster_to_ids, bad_ids, to_delete_ids):
    optimizer = TextGradOptimizer(cfg, memories, memory_stats, log_file_path)
    target_ids_list = list(bad_ids)
    return optimizer.run(target_ids_list, to_delete_ids)