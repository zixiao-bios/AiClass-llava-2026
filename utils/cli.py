"""
通用终端样式工具模块

提供统一的终端彩色输出、对话交互、进度显示等 UI 工具函数，
支持各类 LLM / 多模态模型的命令行交互脚本使用。
"""

import os


# ============ 终端颜色 ============
class Color:
    """ANSI 转义码颜色常量集合。

    在非 TTY 环境下自动禁用颜色输出（如管道、重定向）。

    Attributes:
        CYAN, GREEN, YELLOW 等: 各颜色的 ANSI 转义序列。
        BOLD, DIM, UNDERLINE: 文本样式转义序列。
        END: 重置所有样式的转义序列。
    """
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
        """禁用所有颜色输出，将全部颜色常量设为空字符串。

        用于非 TTY 环境（如管道、CI），避免输出乱码。
        """
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith('_'):
                setattr(cls, attr, "")


# 自动检测标准输出是否为 TTY，非 TTY 时禁用颜色
if not hasattr(os, 'isatty') or not os.isatty(1):
    Color.disable()


# ============ 标题与分隔 ============
def print_header(title="AI 对话助手", width=50):
    """打印居中的标题横幅。

    Args:
        title: 标题文本。
        width: 横幅总宽度（字符数）。
    """
    print(f"\n{Color.CYAN}{'═' * width}")
    print(f"{title:^{width - 6}}")
    print(f"{'═' * width}{Color.END}\n")


def print_divider(char="─", width=50):
    """打印水平分隔线。

    Args:
        char: 分隔线使用的字符。
        width: 分隔线宽度。
    """
    print(f"{Color.GRAY}{char * width}{Color.END}")


# ============ 状态提示 ============
def print_info(msg):
    """打印灰色信息提示。"""
    print(f"{Color.GRAY}{msg}{Color.END}")


def print_success(msg):
    """打印绿色成功提示（带 ✓ 前缀）。"""
    print(f"{Color.GREEN}✓ {msg}{Color.END}")


def print_warning(msg):
    """打印黄色警告提示（带 ⚠ 前缀）。"""
    print(f"{Color.YELLOW}⚠ {msg}{Color.END}")


def print_error(msg):
    """打印红色错误提示（带 ✗ 前缀）。"""
    print(f"{Color.RED}✗ {msg}{Color.END}")


def print_loading(item, label="正在加载"):
    """打印灰色加载提示。

    Args:
        item: 正在加载的对象名称（如模型路径）。
        label: 提示前缀文本。
    """
    print(f"{Color.GRAY}{label}: {item}{Color.END}")


def print_thinking(msg="思考中..."):
    """打印思考中提示，使用回车符实现同行覆盖效果。

    Args:
        msg: 提示文本，下一次输出会覆盖此行。
    """
    print(f"{Color.GRAY}{msg}{Color.END}", end="\r")


# ============ 对话相关 ============
def print_welcome(hints=None):
    """打印对话欢迎信息和操作提示。

    Args:
        hints: 提示列表，默认包含退出和清空命令说明。
    """
    if hints is None:
        hints = ["'quit'/'exit' 退出", "'clear' 清空对话"]
    print(f"{Color.YELLOW}提示: {', '.join(hints)}{Color.END}")
    print_divider()


def print_round(num, label="轮"):
    """打印当前对话轮次标记。

    Args:
        num: 轮次编号。
        label: 轮次单位（默认 "轮"）。
    """
    print(f"\n{Color.BLUE}{Color.BOLD}[第 {num} {label}]{Color.END}")


def print_goodbye(msg="感谢使用，再见！👋"):
    """打印结束语。

    Args:
        msg: 结束语文本。
    """
    print_divider()
    print(f"{Color.CYAN}{msg}{Color.END}\n")


def get_user_prompt(icon="👤", label="用户"):
    """获取用户输入提示符字符串（不打印，供 input() 使用）。

    Args:
        icon: 提示符图标。
        label: 用户标签。

    Returns:
        str: 格式化后的彩色提示符字符串。
    """
    return f"{Color.GREEN}{icon} {label} > {Color.END}"


def format_response(text, icon="🤖", label="助手"):
    """格式化模型回复文本（添加图标和颜色前缀）。

    Args:
        text: 模型的回复文本。
        icon: 助手图标。
        label: 助手标签。

    Returns:
        str: 格式化后的带颜色回复字符串。
    """
    return f"{Color.CYAN}{icon} {label} > {Color.END}{text}"


# ============ 多模态支持 ============
def print_image_info(path):
    """打印已加载图片的文件名。

    Args:
        path: 图片文件路径。
    """
    filename = os.path.basename(path)
    print(f"{Color.MAGENTA}🖼  已加载图片: {filename}{Color.END}")


def print_video_info(path):
    """打印已加载视频的文件名。

    Args:
        path: 视频文件路径。
    """
    filename = os.path.basename(path)
    print(f"{Color.MAGENTA}🎬 已加载视频: {filename}{Color.END}")


def print_audio_info(path):
    """打印已加载音频的文件名。

    Args:
        path: 音频文件路径。
    """
    filename = os.path.basename(path)
    print(f"{Color.MAGENTA}🔊 已加载音频: {filename}{Color.END}")


def print_file_info(path, icon="📄"):
    """打印已加载文件的文件名。

    Args:
        path: 文件路径。
        icon: 文件类型图标。
    """
    filename = os.path.basename(path)
    print(f"{Color.MAGENTA}{icon} 已加载文件: {filename}{Color.END}")


# ============ 进度显示 ============
def print_progress(current, total, prefix="进度", width=30):
    """打印文本进度条（同行覆盖更新）。

    Args:
        current: 当前进度值。
        total: 总数。
        prefix: 进度条前缀标签。
        width: 进度条字符宽度。
    """
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r{Color.BLUE}{prefix}: [{bar}] {percent*100:.1f}%{Color.END}", end="")
    if current >= total:
        print()  # 完成时换行


# ============ 表格显示 ============
def print_kv(key, value, key_width=15):
    """打印对齐的键值对。

    Args:
        key: 键名。
        value: 值。
        key_width: 键名列宽（字符数），用于左对齐。
    """
    print(f"{Color.GRAY}{key:<{key_width}}{Color.END}: {value}")


def print_model_info(info_dict):
    """打印模型信息摘要（自动遍历字典，逐行输出键值对）。

    Args:
        info_dict: 模型信息字典，如 {'参数量': '0.6B', '精度': 'bf16'}。
    """
    print_divider()
    for k, v in info_dict.items():
        print_kv(k, v)
    print_divider()
