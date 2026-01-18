import numpy as np
import math

# ==========================================
# 🔥 BEMR 检索包装器 (修复版)
# ==========================================
class BEMRRetrieverWrapper:
    """
    BEMR 检索包装器：拦截原始检索结果，应用 UCB 公式重排序
    并强制执行截断，解决 FlashRAG 提示词过长问题。
    """
    def __init__(self, original_retriever, memory_stats, cfg):
        self.retriever = original_retriever
        self.memory_stats = memory_stats
        self.cfg = cfg
        self.INIT_VAL = cfg.parameters.INIT_VAL
        
        # 🔥 [修正1] 强制截断阈值 (Small-K)
        # 优先读 parameters.final_topk (3)，读不到就用默认值 3
        if hasattr(cfg, 'parameters'):
            self.final_topk = cfg.parameters.get("final_topk", 3)
        else:
            self.final_topk = 3

        self.lambda1 = cfg.parameters.get('bemr_lambda1', 1.0)
        self.lambda2 = cfg.parameters.get('bemr_lambda2', 0.5)
        
        print(f"🛡️ [Wrapper] BEMR 拦截器就绪 | 最终截断: Top-{self.final_topk}")

    def _calculate_ucb_score(self, doc_id, sim_score):
        stats = self.memory_stats.get(str(doc_id), {'alpha': self.INIT_VAL , 'beta': self.INIT_VAL})
        alpha = stats['alpha']
        beta = stats['beta']
        total = alpha + beta
        
        mean_utility = alpha / total
        exploration = math.sqrt(math.log(max(total, 1)) / total)
        
        # --- 核心公式 ---
        # Part A: UCB 主导部分 (效用 + 探索)
        ucb_part = (self.lambda1 * mean_utility) + (self.lambda2 * exploration)
        
        # Part B: BM25 微弱影响 (Tie-Breaker)
        # 0.001 的权重足以在 UCB 相同时区分高下，但不足以让 BM25 干扰 UCB 的判断
        bm25_part = 0.001 * sim_score
        
        final_score = ucb_part + bm25_part
        return final_score, ucb_part  # 🔥 返回两个值，方便 Debug 显示纯 UCB 分数

    # 🔥 [修正2] 签名必须完全匹配 FlashRAG 的 Retriever 接口
    def search(self, query_list, num=None, return_score=False):
        # ==========================================
        # 1. 确定海选数量 (Funnel Stage 1)
        # ==========================================
        # 即使 FlashRAG 只我们要 3 条，我们也要先抓 20 条回来挑！
        INITIAL_POOL_SIZE = 20 
        search_k = max(num if num else 0, INITIAL_POOL_SIZE)
        
        # 2. 调用底层 batch_search
        raw_output = self.retriever.batch_search(query_list, num=search_k, return_score=True)
        
        if isinstance(raw_output, tuple):
            batch_hits, batch_scores = raw_output
        else:
            batch_hits = raw_output
            batch_scores = [[0.0] * len(h) for h in batch_hits]

        reranked_results = []
        reranked_scores = []

        # 遍历每一个 Query
        for q_idx, (hit_list, score_list) in enumerate(zip(batch_hits, batch_scores)):
            
            # --- 📊 [Debug 准备] ---
            debug_info = [] 
            
            # 归一化准备
            if not score_list:
                reranked_results.append([])
                reranked_scores.append([])
                continue
                
            min_s, max_s = min(score_list), max(score_list)
            denominator = max_s - min_s if (max_s - min_s) > 1e-6 else 1.0
            
            scored_hits = []
            
            # --- 循环处理 20 个候选记忆 ---
            for i, hit in enumerate(hit_list):
                doc_id = hit.get('id')
                raw_bm25 = score_list[i] 
                
                # 归一化 BM25 (0~1)
                norm_bm25 = (raw_bm25 - min_s) / denominator
                
                # 计算分数 (获取 Final 和 纯 UCB)
                final_score, pure_ucb = self._calculate_ucb_score(doc_id, norm_bm25)
                
                # 写入新分数
                hit['score'] = final_score
                scored_hits.append(hit)
                
                # 获取状态用于展示
                stats = self.memory_stats.get(str(doc_id), {'alpha': self.INIT_VAL, 'beta': self.INIT_VAL})
                
                # 存入 Debug 列表
                debug_info.append({
                    "id": doc_id,
                    "bm25_raw": raw_bm25,
                    "bm25_norm": norm_bm25,
                    "pure_ucb": pure_ucb,    # 纯 UCB 分数 (不含 0.001*BM25)
                    "final_score": final_score, # 最终排序依据
                    "stats": f"{stats['alpha']:.1f}/{stats['beta']:.1f}"
                })

            # --- 排序 ---
            # 按 Final Score 降序
            scored_hits.sort(key=lambda x: x['score'], reverse=True)
            
            # --- 截断 (Top-K) ---
            cutoff = self.final_topk
            if num and num < self.final_topk:
                cutoff = num
            truncated_hits = scored_hits[:cutoff]
            truncated_scores = [h['score'] for h in truncated_hits]
            
            reranked_results.append(truncated_hits)
            reranked_scores.append(truncated_scores)

            # ==========================================
            # 🕵️‍♂️ [显微镜] 打印详细排位表
            # ==========================================
            # print(f"\n🔎 [Query {q_idx+1}] 检索详情监控 (Top-{search_k} -> Top-{cutoff})")
            # # 表头格式化：增加了 Pure UCB 和 Final Score
            # print(f"{'Rank':<5} | {'ID':<6} | {'BM25(Raw)':<10} | {'Status(A/B)':<12} | {'Pure UCB':<10} | {'Final Score':<11} | {'Result'}")
            # print("-" * 88)
            
            # 必须让 debug_info 也按 Final Score 排序，才能和 Rank 对应上
            debug_info.sort(key=lambda x: x['final_score'], reverse=True)
            
            # for rank, info in enumerate(debug_info):
            #     is_selected = "✅ PICK" if rank < cutoff else "❌ DROP"
                
            #     # 打印前5名 和 后2名
            #     if rank < 20 or rank >= len(debug_info) - 2: 
            #         print(f"{rank+1:<5} | {info['id']:<6} | {info['bm25_raw']:<10.2f} | {info['stats']:<12} | {info['pure_ucb']:<10.4f} | {info['final_score']:<11.4f} | {is_selected}")
            #     elif rank == 5:
            #         print(f"{'...':<5} | {'...':<6} | {'...':<10} | {'...':<12} | {'...':<10} | {'...':<11} | ...")
            # print("=" * 88)
            # # ==========================================

        if return_score:
            return reranked_results, reranked_scores
        else:
            return reranked_results
        
    def batch_search(self, query_list, num=None, return_score=False):
        """
        强制重定向 batch_search 到我们需要执行 UCB 逻辑的 search 方法
        """
        print(f"✅ [Wrapper] 成功拦截 batch_search请求，转入 BEMR 逻辑处理...")
        return self.search(query_list, num, return_score)

    def __getattr__(self, name):
        # 建议保留这个 print 用于监控未来是否还有其他方法泄露
        print(f"⚠️ [Wrapper Bypass] 正在透传方法: {name}")
        return getattr(self.retriever, name)