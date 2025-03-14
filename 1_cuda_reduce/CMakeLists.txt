find_package(CUDA REQUIRED)

add_executable(my_reduce_1 my_reduce_1.cu)
target_link_libraries(my_reduce_1 PRIVATE  ${CUDA_cublas_LIBRARY})

add_executable(my_reduce_shuffle my_reduce_shuffle.cu)
target_link_libraries(my_reduce_shuffle PRIVATE  ${CUDA_cublas_LIBRARY})