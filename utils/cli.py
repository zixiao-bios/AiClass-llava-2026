"""通用终端样式工具函数 - 支持各类LLM/多模态模型"""

import os

# ============ 终端颜色 ============
class Color:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    @classmethod
    def disable(cls):
        """禁用颜色输出（用于非TTY环境）"""
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith('_'):
                setattr(cls, attr, "")

# 自动检测是否支持颜色
if not hasattr(os, 'isatty') or not os.isatty(1):
    Color.disable()

# ============ 标题与分隔 ============
def print_header(title="AI 对话助手", width=50):
    """打印标题头"""
    print(f"\n{Color.CYAN}{'═' * width}")
    print(f"{title:^{width - 6}}")
    print(f"{'═' * width}{Color.END}\n")

def print_divider(char="─", width=50):
    """打印分隔线"""
    print(f"{Color.GRAY}{char * width}{Color.END}")

# ============ 状态提示 ============
def print_info(msg):
    """信息提示"""
    print(f"{Color.GRAY}{msg}{Color.END}")

def print_success(msg):
    """成功提示"""
    print(f"{Color.GREEN}✓ {msg}{Color.END}")

def print_warning(msg):
    """警告提示"""
    print(f"{Color.YELLOW}⚠ {msg}{Color.END}")

def print_error(msg):
    """错误提示"""
    print(f"{Color.RED}✗ {msg}{Color.END}")

def print_loading(item, label="正在加载"):
    """加载提示"""
    print(f"{Color.GRAY}{label}: {item}{Color.END}")

def print_thinking(msg="思考中..."):
    """思考中提示（同行覆盖）"""
    print(f"{Color.GRAY}{msg}{Color.END}", end="\r")

# ============ 对话相关 ============
def print_welcome(hints=None):
    """打印欢迎提示"""
    if hints is None:
        hints = ["'quit'/'exit' 退出", "'clear' 清空对话"]
    print(f"{Color.YELLOW}提示: {', '.join(hints)}{Color.END}")
    print_divider()

def print_round(num, label="轮"):
    """打印对话轮次"""
    print(f"\n{Color.BLUE}{Color.BOLD}[第 {num} {label}]{Color.END}")

def print_goodbye(msg="感谢使用，再见！👋"):
    """打印结束语"""
    print_divider()
    print(f"{Color.CYAN}{msg}{Color.END}\n")

def get_user_prompt(icon="👤", label="用户"):
    """获取用户输入提示符"""
    return f"{Color.GREEN}{icon} {label} > {Color.END}"

def format_response(text, icon="🤖", label="助手"):
    """格式化模型回复"""
    return f"{Color.CYAN}{icon} {label} > {Color.END}{text}"

# ============ 多模态支持 ============
def print_image_info(path):
    """显示图片信息"""
    filename = os.path.basename(path)
    print(f"{Color.MAGENTA}🖼  已加载图片: {filename}{Color.END}")

def print_video_info(path):
    """显示视频信息"""
    filename = os.path.basename(path)
    print(f"{Color.MAGENTA}🎬 已加载视频: {filename}{Color.END}")

def print_audio_info(path):
    """显示音频信息"""
    filename = os.path.basename(path)
    print(f"{Color.MAGENTA}🔊 已加载音频: {filename}{Color.END}")

def print_file_info(path, icon="📄"):
    """显示文件信息"""
    filename = os.path.basename(path)
    print(f"{Color.MAGENTA}{icon} 已加载文件: {filename}{Color.END}")

# ============ 进度显示 ============
def print_progress(current, total, prefix="进度", width=30):
    """打印进度条"""
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r{Color.BLUE}{prefix}: [{bar}] {percent*100:.1f}%{Color.END}", end="")
    if current >= total:
        print()

# ============ 表格显示 ============
def print_kv(key, value, key_width=15):
    """打印键值对"""
    print(f"{Color.GRAY}{key:<{key_width}}{Color.END}: {value}")

def print_model_info(info_dict):
    """打印模型信息"""
    print_divider()
    for k, v in info_dict.items():
        print_kv(k, v)
    print_divider()
