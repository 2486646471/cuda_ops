find_package(CUDA REQUIRED)

add_executable(my_softmax my_softmax.cu)
target_link_libraries(my_softmax PRIVATE  ${CUDA_cublas_LIBRARY})
