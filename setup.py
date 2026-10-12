# SPDX-License-Identifier: Apache-2.0
"""Build the lowlat_gemm CUDA extension.

Environment:
  LLG_CUDA_ARCHS   semicolon-separated SM targets (default "120a"; e.g. "120a;121a")
  LLG_PTXAS_VERBOSE=1   print per-kernel register/spill usage from ptxas
"""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

archs = [a.strip() for a in os.environ.get("LLG_CUDA_ARCHS", "120a").split(";") if a.strip()]
# Arch-specific targets ("120a", "121a") are required for the FP8 / NVFP4 MMAs and the e2m1 conversions; their PTX
# does not carry forward, so only SASS is embedded for them.
gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]
gencode += [f"-gencode=arch=compute_{a},code=compute_{a}" for a in archs if not a.endswith("a")]

# conda-packaged toolkits keep headers and libraries under targets/<arch>-linux
include_dirs, library_dirs = [], []
if CUDA_HOME:
    for sub in ("targets/x86_64-linux", "targets/sbsa-linux"):
        root = os.path.join(CUDA_HOME, sub)
        if os.path.isdir(os.path.join(root, "include")):
            include_dirs.append(os.path.join(root, "include"))
            library_dirs.append(os.path.join(root, "lib"))

nvcc_flags = ["-O3", "-std=c++20", "-lineinfo", "--expt-relaxed-constexpr", *gencode]
if os.environ.get("LLG_PTXAS_VERBOSE") == "1":
    nvcc_flags.append("--ptxas-options=-v")

setup(
    ext_modules=[
        CUDAExtension(
            name="lowlat_gemm._C",
            sources=[
                "csrc/ops.cpp",
                "csrc/gemv_lpr8.cu",
                "csrc/gemv_lpr16.cu",
                "csrc/gemv_lpr32.cu",
                "csrc/quant.cu",
                "csrc/tc_info.cu",
                *sorted(str(p) for p in Path("csrc").glob("tc_*.cu") if p.name != "tc_info.cu"),
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args={"cxx": ["-O3", "-std=c++20"], "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
