find_package(CUDA REQUIRED)

add_executable(my_add_1 my_add_1.cu)
target_link_libraries(my_add_1 PRIVATE  ${CUDA_cublas_LIBRARY})

add_executable(my_sigmoid my_sigmoid.cu)
target_link_libraries(my_sigmoid PRIVATE  ${CUDA_cublas_LIBRARY})

add_executable(my_relu my_relu.cu)
target_link_libraries(my_relu PRIVATE  ${CUDA_cublas_LIBRARY})
