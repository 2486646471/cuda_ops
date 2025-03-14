#设置最低版本
cmake_minimum_required(VERSION 3.16.0)

#项目信息
project(cuda_project VERSION 1.0 LANGUAGES CUDA C CXX)

add_subdirectory(1_cuda_reduce)

add_subdirectory(2_cuda_segmm)

add_subdirectory(3_cuda_elementwise)