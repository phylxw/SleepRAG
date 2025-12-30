import os
import sys
# ==============================================================================
# 🔥 [功能增强] 日志记录器 (同时输出到终端和文件)
# ==============================================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 实时刷新，防止程序崩了没保存

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logging(root_dir, pipeline_id):
    """设置全局日志重定向"""
    log_dir = os.path.join(root_dir, "results/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{pipeline_id}.log")
    
    # 劫持 stdout 和 stderr
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout # 把错误也打到同一个文件里
    
    print(f"📝 全局日志已开启，保存至: {log_file}")
    return log_file