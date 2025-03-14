find_package(CUDA REQUIRED)

add_executable(my_segmm_1 my_segmm_1.cu)
target_link_libraries(my_segmm_1 PRIVATE  ${CUDA_cublas_LIBRARY})

add_executable(my_segmm_2 my_segmm_2.cu)
target_link_libraries(my_segmm_2 PRIVATE  ${CUDA_cublas_LIBRARY})
