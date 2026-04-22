from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


ext_modules = [
    Pybind11Extension(
        "hppca.hppca_bindings",
        ["src/hppca/bind/hppca_bindings.cc"],
        libraries=["openblas", "lapacke"],
        extra_compile_args=["-O3"],
    )
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
