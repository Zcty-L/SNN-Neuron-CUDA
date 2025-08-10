#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

//#include "test.cuh"
//#include "op/softmax/softmax.cuh"
//#include "op/neuron/lif.cuh"
// #include "op/conv2d.h"
#include "op/gemm.cuh"
#include "op/linear/linear.cuh"

int main(int argc, char **argv)
{
    const int DEVICE_ID = 0;
    cudaSetDevice(DEVICE_ID);

    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, DEVICE_ID);
    std::cout << "DEVICE  ID: " << DEVICE_ID << "\t" << deviceProp.name << std::endl; // 4060 Laptop GPU
    std::cout << "        SM: " << deviceProp.multiProcessorCount << std::endl; // 24
    std::cout << "Shared Mem: " << deviceProp.sharedMemPerBlock << " KB" << std::endl; // 49152 KB
    std::cout << "regs Block: " << deviceProp.regsPerBlock << std::endl; // 65536

    // test_shfl_down_sync();
    // test_shfl_xor_sync();
    // softmax_main();
    // lif_main();
    // conv_5_weightsT_main();

    // gemm_main();
    linear_main();

    return 0;
}

