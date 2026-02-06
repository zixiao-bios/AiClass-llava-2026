#!/usr/bin/env python
"""检测本地环境，输出对应的 flash-attn 2.8.3 wheel 文件名"""

import sys
import platform

try:
    import torch
except ImportError:
    print("错误: PyTorch 未安装")
    sys.exit(1)

# Python 标签
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

# PyTorch 版本
torch_version = ".".join(torch.__version__.split("+")[0].split(".")[:2])

# CUDA 版本
cuda_major = torch.version.cuda.split(".")[0] if torch.cuda.is_available() else None
if not cuda_major:
    print("错误: CUDA 不可用")
    sys.exit(1)

# CXX11 ABI
abi_tag = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"

# 平台
system = platform.system().lower()
machine = platform.machine().lower()
if system == "linux" and machine == "x86_64":
    platform_tag = "linux_x86_64"
elif system == "windows" and "64" in machine:
    platform_tag = "win_amd64"
else:
    platform_tag = f"{system}_{machine}"

# 输出链接
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu{cuda_major}torch{torch_version}cxx11abi{abi_tag}-{py_tag}-{py_tag}-{platform_tag}.whl")
