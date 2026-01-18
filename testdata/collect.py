import json
import os

def clean_memory_stats(input_path,output_path):
    # 1. 构造输出文件名 (在原文件名后加 _cleaned)
    print(f"📂 正在读取文件: {input_path}")

    if not os.path.exists(input_path):
        print("❌ 文件不存在，请检查路径！")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_count = len(data)
        new_data = {}
        deleted_count = 0

        # 2. 核心过滤逻辑
        # 目标: 删除 alpha == 0.5 AND beta == 0.5 的条目
        TARGET_VAL = 0.5 

        for mid, stats in data.items():
            alpha = stats.get("alpha", 0)
            beta = stats.get("beta", 0)

            # 如果两者都等于目标值 (说明是没动过的初始值/僵尸值)，则跳过
            if alpha == TARGET_VAL and beta == TARGET_VAL:
                deleted_count += 1
                continue
            
            # 否则保留
            new_data[mid] = stats

        # 3. 保存新文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

        print("-" * 40)
        print(f"📊 处理完成！")
        print(f"   - 原始条目数: {original_count}")
        print(f"   - ✂️ 移除条目 (全{TARGET_VAL}): {deleted_count}")
        print(f"   - ✅ 保留条目数: {len(new_data)}")
        print(f"💾 新文件已保存至: {output_path}")
        print("-" * 40)

    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    # 这里填你的文件绝对路径
    input_path = "/root/workspace/jychen/ex/collects/hmmtex/json/hmmtex_memory_stats.json"
    output_path = "/root/workspace/jychen/ex/testdata/stats.json"
    clean_memory_stats(input_path,output_path)