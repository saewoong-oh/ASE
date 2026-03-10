from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "ase_core",
        sources=[
            "cpp/dft_engine.cpp",
            "cpp/bindings.cpp",
        ],
        include_dirs=["cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="auto_sound_engineer",
    version="1.0.0",
    description="Automated Sound Engineer with C++ DFT core",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=[],
    python_requires=">=3.9",
)