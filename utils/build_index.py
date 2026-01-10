from datasets import load_dataset
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf
import os
import json
import tqdm
import bm25s

def build_index(corpus_file: str, index_dir: str):
    """构建 BM25 索引"""

    print(f"🔨 [Index] 正在为 {corpus_file} 构建 BM25 索引...")
    corpus_texts = []
    
    # 使用 bm25s 库
    with open(corpus_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                # 🔥 [核心修改] 字段兼容逻辑
                # 优先找 'contents'，没有就找 'question' (MBPP normalized)，再没有找 'prompt' 或 'text'
                content = item.get('contents') or item.get('question') or item.get('prompt') or item.get('text')
                
                if content:
                    corpus_texts.append(content)
                else:
                    # 如果实在找不到，打印警告但不崩溃（或者你可以选择抛出异常）
                    print(f"⚠️ [Line {i}] 警告：未找到有效文本字段 (contents/question/prompt)，已跳过。Keys: {list(item.keys())}")
            except json.JSONDecodeError:
                print(f"⚠️ [Line {i}] JSON 解析失败，跳过。")
                continue

    if not corpus_texts:
        raise ValueError(f"❌ 索引构建失败：{corpus_file} 中没有提取到任何有效文本！请检查字段映射。")

    print(f"📊 提取到 {len(corpus_texts)} 条文本，开始分词...")
    
    # 后面保持不变
    corpus_tokens = bm25s.tokenize(corpus_texts)
    retriever_builder = bm25s.BM25()
    retriever_builder.index(corpus_tokens)
    retriever_builder.save(index_dir)
    
    # FlashRAG 要求的额外文件
    with open(os.path.join(index_dir, "stopwords.tokenizer.json"), "w") as f:
        json.dump([], f)
    with open(os.path.join(index_dir, "vocab.tokenizer.json"), "w") as f:
        vocab = corpus_tokens.vocab
        # 兼容性处理
        json.dump({"word_to_id": vocab, "stem_to_sid": vocab, "word_to_stem": {k: k for k in vocab}}, f)
    print("✅ 索引构建完成！")