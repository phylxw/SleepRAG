import os
import json
import hydra
from omegaconf import DictConfig

def parse_wrong_ids_from_log(log_path):
    """从单个日志文件中提取 Result 为 ❌ Wrong 的 ID"""
    print(f"📖 正在扫描日志: {log_path}")
    if not os.path.exists(log_path):
        print(f"   ⚠️ 文件不存在，跳过: {log_path}")
        return set()

    wrong_ids = set()
    current_id = None
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # 1. 捕获 ID
            if line.startswith("[ID]:"):
                try:
                    # 格式: [ID]: 1160
                    current_id = line.split(":", 1)[1].strip()
                except:
                    current_id = None
            
            # 2. 捕获 结果
            elif line.startswith("[Result]:"):
                if "Wrong" in line or "❌" in line:
                    if current_id is not None:
                        wrong_ids.add(current_id)
                # 重置 ID，防止错位
                current_id = None
                
    print(f"   found {len(wrong_ids)} 个错题。")
    return wrong_ids

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. 读取配置
    input_logs = cfg.wrong_set.input_logs
    source_file = cfg.wrong_set.source_test_file
    output_file = cfg.wrong_set.output_file
    
    # 如果配置里只是单个字符串，转为列表
    if isinstance(input_logs, str):
        input_logs = [input_logs]

    # 2. 收集所有错题 ID (自动去重)
    all_wrong_ids = set()
    for log_file in input_logs:
        ids = parse_wrong_ids_from_log(log_file)
        all_wrong_ids.update(ids)
    
    print(f"\n🚫 总共发现 {len(all_wrong_ids)} 个唯一的错题 ID。")
    if len(all_wrong_ids) == 0:
        print("🎉 没有发现错题，或者日志路径配置错误。")
        return

    # 3. 从源文件提取题目内容
    print(f"\n🔍 正在从源数据 {source_file} 中提取题目内容...")
    
    if not os.path.exists(source_file):
        # 尝试自动修正路径：如果 config 里是硬编码的 AMATH，但实际只有 MATH
        # 这里做一个简单的容错，或者直接报错
        print(f"❌ 源测试集文件未找到: {source_file}")
        print("   请检查 pre.py 生成的 _test_data.jsonl 文件名是否与 config 中一致。")
        return

    wrong_entries = []
    found_ids = set()
    
    with open(source_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                mid = str(item['id'])
                
                if mid in all_wrong_ids:
                    wrong_entries.append(item)
                    found_ids.add(mid)
            except json.JSONDecodeError:
                continue
    
    # 4. 保存错题集
    missing_ids = all_wrong_ids - found_ids
    if missing_ids:
        print(f"⚠️ 警告: 有 {len(missing_ids)} 个ID在源文件中没找到 (可能是源文件版本不匹配): {list(missing_ids)[:5]}...")

    print(f"💾 正在保存 {len(wrong_entries)} 条错题到: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in wrong_entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("✅ 错题集生成完成！")
    print(f"🚀 下次运行 Eval 时，请使用: python eval.py paths.root={os.path.dirname(output_file)} +experiment.test_file_override={os.path.basename(output_file)}")

if __name__ == "__main__":
    main()