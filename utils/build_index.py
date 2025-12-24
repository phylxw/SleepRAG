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
        for line in f:
            corpus_texts.append(json.loads(line)['contents'])
    
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