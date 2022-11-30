from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='suds_cuda',
    ext_modules=[
        CUDAExtension('suds_cuda', [
            'suds/cpp/suds_cpp.cpp',
            'suds/cpp/suds_cuda.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
