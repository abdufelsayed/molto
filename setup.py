import os
import sys
from pathlib import Path

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel

CUSTOM_KERNEL_FLAG = "--with-custom-kernel"
TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_CUSTOM_KERNEL_DEPLOYMENT_TARGET = "15.0"


def _with_custom_kernel() -> bool:
    if CUSTOM_KERNEL_FLAG in sys.argv:
        sys.argv.remove(CUSTOM_KERNEL_FLAG)
        return True
    return os.environ.get("OMLX_WITH_CUSTOM_KERNEL", "").strip().lower() in TRUTHY


def _custom_kernel_build_kwargs() -> dict:
    if not _with_custom_kernel():
        return {}

    target = (
        os.environ.get("OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET")
        or os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        or DEFAULT_CUSTOM_KERNEL_DEPLOYMENT_TARGET
    )
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", target)
    cmake_args = os.environ.get("CMAKE_ARGS", "").strip()
    if "CMAKE_OSX_DEPLOYMENT_TARGET" not in cmake_args:
        target_arg = f"-DCMAKE_OSX_DEPLOYMENT_TARGET={target}"
        os.environ["CMAKE_ARGS"] = (
            f"{cmake_args} {target_arg}".strip() if cmake_args else target_arg
        )
        cmake_args = os.environ["CMAKE_ARGS"]

    # CMake otherwise chooses the first framework Python on PATH, which can
    # differ from the interpreter running pip (and lack nanobind / MLX).  The
    # extensions must use the active environment's ABI and CMake packages.
    python_args = " ".join(
        (
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        )
    )
    if "Python_EXECUTABLE" not in cmake_args:
        os.environ["CMAKE_ARGS"] = f"{cmake_args} {python_args}".strip()

    from mlx import extension

    return {
        "ext_modules": [
            extension.CMakeExtension(
                "omlx.custom_kernels.bonsai._ext",
                sourcedir="omlx/custom_kernels/bonsai/csrc",
            ),
            extension.CMakeExtension(
                "omlx.custom_kernels.decode_fast._ext",
                sourcedir="omlx/custom_kernels/decode_fast/csrc",
            ),
            extension.CMakeExtension(
                "omlx.custom_kernels.glm_moe_dsa._ext",
                sourcedir="omlx/custom_kernels/glm_moe_dsa/csrc",
            ),
            extension.CMakeExtension(
                "omlx.custom_kernels.minimax_m3._ext",
                sourcedir="omlx/custom_kernels/minimax_m3/csrc",
            ),
            extension.CMakeExtension(
                "omlx.custom_kernels.qwen35_prefill._ext",
                sourcedir="omlx/custom_kernels/qwen35_prefill/csrc",
            ),
        ],
        "cmdclass": {"build_ext": extension.CMakeBuild},
    }


class DashboardWheel(bdist_wheel):
    """Node makes the bundle platform specific even without custom kernels."""

    def finalize_options(self):
        super().finalize_options()
        if (Path(__file__).parent / "omlx/_dashboard/runtime/node").is_file():
            self.root_is_pure = False
            self.plat_name = "macosx_15_0_arm64"
            self.plat_name_supplied = True

    def get_tag(self):
        python, abi, platform = super().get_tag()
        if not self.distribution.has_ext_modules() and not self.root_is_pure:
            return "py3", "none", platform
        return python, abi, platform


if __name__ == "__main__":
    kwargs = _custom_kernel_build_kwargs()
    kwargs.setdefault("cmdclass", {})["bdist_wheel"] = DashboardWheel
    setup(**kwargs)
