find_package(PythonInterp REQUIRED)
set(PYTHON_DIR D:/Resource/Conda/envs/py39)
message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_VERSION_STRING: ${PYTHON_VERSION_STRING}")

cmake_minimum_required(VERSION 3.18)
project(CUDATest LANGUAGES CUDA CXX)
#project(CUDATest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

set(CMAKE_CUDA_ARCHITECTURES 86) # 4090_89 4060_86


# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif ()


if (WIN32)
    message(STATUS "Windows")
    option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

    # cuda
    include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include")
    link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64")

elseif (UNIX)
    message(STATUS "Linux")
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    enable_language(CUDA)

    # cuda
    include_directories("/usr/local/cuda/include")
    link_directories("/usr/local/cuda/lib64")

    # cudnn
    include_directories("/home/li/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include")
    link_directories("/home/li/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib")

endif ()

set(TORCH_DIR D:/Resource/Conda/envs/py39/Lib/site-packages/torch)
set(PYTHON_DIR D:/Resource/Conda/envs/py39)


# cuda
#set(USE_CUDNN 1)
#set(CAFFE2_USE_CUDNN 1)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})
message(STATUS "CUDA VERSION     : ${CUDA_VERSION_STRING}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES   : ${CUDA_LIBRARIES}")

# torch
set(CMAKE_PREFIX_PATH ${TORCH_DIR}/share/cmake/Torch)
find_package(Torch REQUIRED)
link_directories(${TORCH_INSTALL_PREFIX}/lib)
message(STATUS "     TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "  TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/test
        ${PROJECT_SOURCE_DIR}/op

        ${PYTHON_DIR}/include

        ${TORCH_INCLUDE_DIRS}
)


# test cu
set(TEST_SRC ${PROJECT_SOURCE_DIR}/test/test.cu)

# softmax
set(SOFTMAX_SRC ${PROJECT_SOURCE_DIR}/op/softmax/softmax.cu)

# neuron
file(GLOB NEURON_SRC      ${PROJECT_SOURCE_DIR}/op/neuron/*.cu ${PROJECT_SOURCE_DIR}/op/neuron/neuron_torch_api.hpp)
#file(GLOB NEURON_LITE_SRC ${PROJECT_SOURCE_DIR}/op/neuron_lite/*.cu ${PROJECT_SOURCE_DIR}/op/neuron_lite/neuron_torch_api.hpp)
message(STATUS "     NEURON_SRC: ${NEURON_SRC}")
#message(STATUS "NEURON_LITE_SRC: ${NEURON_LITE_SRC}")

# conv
file(GLOB Conv2D_SRC ${PROJECT_SOURCE_DIR}/op/conv/*.cu)
message(STATUS "Conv2D_SRC: ${Conv2D_SRC}")

# linear
file(GLOB Linear_SRC ${PROJECT_SOURCE_DIR}/op/linear/*.cu)
message(STATUS "Linear_SRC: ${Linear_SRC}")

# GEMM
file(GLOB GeMM_SRC ${PROJECT_SOURCE_DIR}/op/gemm.cu)
message(STATUS "GeMM_SRC: ${GeMM_SRC}")

# pybind
set(PYBIND_SRC ${PROJECT_SOURCE_DIR}/op/neuron_pybind.cpp)

if (WIN32)
    add_executable(
            main main.cpp
#            ${TEST_SRC} ${NEURON_SRC} ${PYBIND_SRC}
#            ${SOFTMAX_SRC}
#            ${Conv2D_SRC}
            ${Linear_SRC}
            ${GeMM_SRC}
            ${NEURON_SRC}
#            ${NEURON_LITE_SRC}
#            ${PROJECT_SOURCE_DIR}/op/utils.cpp
    )
    target_compile_definitions(main PUBLIC MY_MACRO=233)
elseif (UNIX)
    add_executable(main main.cpp ${TEST_SRC})
endif ()

target_link_libraries(main ${TORCH_LIBRARIES})
target_link_libraries(main cuda cublas cudart cudnn)
target_link_directories(main PRIVATE ${TORCH_DIR}/lib)
target_link_directories(main PUBLIC ${TORCH_DIR}/lib)

set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET main POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} $<TARGET_FILE_DIR:main>)

    file(GLOB CONDA_DLL_DIR "D:/Resource/Conda/envs/py39/Library/bin/*.dll")
    add_custom_command(TARGET main POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CONDA_DLL_DIR} $<TARGET_FILE_DIR:main>)
endif (MSVC)

