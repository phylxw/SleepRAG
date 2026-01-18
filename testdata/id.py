import json
import os
import shutil

def safe_rename_memory_id(file_path):
    print(f"📂 正在读取: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ 文件不存在")
        return

    # 1. 创建一个临时的输出文件 .tmp
    temp_path = file_path + ".tmp"
    count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f_in, \
             open(temp_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line = line.strip()
                if not line: continue
                
                try:
                    item = json.loads(line)
                    
                    # 🔥 核心修改：memory_id -> id
                    if "memory_id" in item:
                        val = item.pop("memory_id")
                        # 新建一个字典把 id 放最前面 (可选)
                        new_item = {"id": val}
                        new_item.update(item)
                        item = new_item
                    
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
                except json.JSONDecodeError:
                    pass
        
        # 2. 只有在成功写入后，才覆盖源文件
        if count > 0:
            shutil.move(temp_path, file_path) # 这里的 move 会覆盖原文件
            print("-" * 40)
            print(f"✅ 成功原地修改！共处理 {count} 行")
            print(f"💾 文件已更新: {file_path}")
        else:
            print("⚠️ 读取行数为 0，未修改原文件（请检查源文件是否为空）")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    file_path = "/root/workspace/jychen/ex/testdata/cluster_output.jsonl"
    safe_rename_memory_id(file_path)