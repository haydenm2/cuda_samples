cmake_minimum_required(VERSION 3.0.2)
project(cuda_samples)

set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)

include_directories(
#   include
)

cuda_add_executable(vectorAdd src/vectorAdd.cu)
cuda_add_executable(matrixAdd src/matrixAdd.cu)
