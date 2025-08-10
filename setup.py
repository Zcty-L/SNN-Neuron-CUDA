import glob
import os.path as osp
from os.path import join as pjoin
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
print("ROOT_DIR: ", ROOT_DIR)

"""
env:
    cuda              12.4
    cupy-cuda12x      13.4.1
    python            3.11.11             
    pytorch           2.5.1          
    pytorch-cuda      12.4
    spikingjelly      latest
"""

setup(
    name='spike_cuda_op',
    version='1.0',
    author='LI',
    author_email='12433002@mail.sustech.edu.cn',
    description='spike_cuda_op',
    long_description='spike_cuda_op',
    ext_modules=[
        CUDAExtension(
            name='spike_cuda_op',
            sources=[
                pjoin("op", "neuron", "if.cu"),
                pjoin("op", "neuron", "lif.cu"),
                pjoin("op", "neuron", "plif.cu"),
                pjoin("op", "neuron", "cds_lif.cu"),
                pjoin("op", "neuron_pybind.cpp"),
            ],
            include_dirs=[
                ROOT_DIR,
                pjoin("op", "neuron", "neuron.h"),
                pjoin("op", "neuron", "neuron_torch_api.hpp"),
            ],
            extra_compile_args={
                'cxx': [
                    '-O2', "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_HALF_OPERATORS__"
                ],
                'nvcc': [
                    '-O2', "-arch=sm_86",
                    "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_HALF_OPERATORS__"
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
