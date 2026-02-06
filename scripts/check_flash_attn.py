#!/usr/bin/env python
"""
Flash Attention wheel 下载链接生成器

检测当前环境的 Python 版本、PyTorch 版本、CUDA 版本和平台架构，
自动拼接出对应的 flash-attn 2.8.3 预编译 wheel 下载地址。
"""

import sys
import platform

try:
    import torch
except ImportError:
    print("错误: PyTorch 未安装")
    sys.exit(1)

# ---- 收集环境信息 ----

# Python 版本标签，如 "cp310"（CPython 3.10）
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

# PyTorch 主次版本号，如 "2.4"（去掉 patch 和 cuda 后缀）
torch_version = ".".join(torch.__version__.split("+")[0].split(".")[:2])

# CUDA 主版本号，如 "12"
cuda_major = torch.version.cuda.split(".")[0] if torch.cuda.is_available() else None
if not cuda_major:
    print("错误: CUDA 不可用")
    sys.exit(1)

# C++ ABI 标签：PyTorch 编译时的 _GLIBCXX_USE_CXX11_ABI 设置
abi_tag = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"

# 操作系统和 CPU 架构标签
system = platform.system().lower()
machine = platform.machine().lower()
if system == "linux" and machine == "x86_64":
    platform_tag = "linux_x86_64"
elif system == "windows" and "64" in machine:
    platform_tag = "win_amd64"
else:
    platform_tag = f"{system}_{machine}"

# ---- 输出完整下载链接 ----
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
      f"flash_attn-2.8.3+cu{cuda_major}torch{torch_version}cxx11abi{abi_tag}"
      f"-{py_tag}-{py_tag}-{platform_tag}.whl")
