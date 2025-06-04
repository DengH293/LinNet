import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars


(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="linops",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["linops"],
    package_dir={"linops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="linops_cuda",
            sources=[
                os.path.join('src', 'insert_hashtable_cuda.cpp'),
                os.path.join('src', 'insert_hashtable_cuda_kernel.cu'),
                os.path.join('src', 'linear_voxel_query_cuda.cpp'),
                os.path.join('src', 'linear_voxel_query_cuda_kernel.cu'),
                os.path.join('src', 'linearprobing.cu'),
                os.path.join('src', 'linear_ops_api.cpp'),
            ],
            extra_compile_args={
                "cxx": ["-g", f"-I{os.path.join(this_dir, 'src')}"],
                "nvcc": ["-O2", f"-I{os.path.join(this_dir, 'src')}"],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
