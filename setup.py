from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='blockwise_sparse',
    ext_modules=[
        CUDAExtension(
            name='blockwise_sparse',
            sources=['blockwise_multiply_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
