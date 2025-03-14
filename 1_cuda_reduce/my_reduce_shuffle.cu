#include <stdio.h>
#include <cuda_runtime.h>
#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

template<int num_per_block>
__global__ void reduce(float* d_input, float* d_output) {
    float sum = 0.f;
    int tid = threadIdx.x;
    float* d_input_start = d_input + blockIdx.x * num_per_block;

    for (int i = 0; i < num_per_block / THREAD_PER_BLOCK; i++) {
        sum += d_input_start[tid + i * THREAD_PER_BLOCK];
    }
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    for (int s = 16; s > 0; s >>= 1) {
        if (lane_id < s) {
            sum += __shfl_down_sync(0xffffffff, sum, s);
        }
    }

    __shared__ float warpSums[32];
    if (lane_id == 0) {
        warpSums[warp_id] = sum;
    } 
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < blockDim.x / 32) ? warpSums[lane_id] : 0.f;
        for (int s = 16; s > 0; s >>= 1) {
            if (lane_id < s) {
                sum += __shfl_down_sync(0xffffffff, sum, s);
            }
        }
    }
    // 写入结果
    if (tid == 0) {
        d_output[blockIdx.x] = warpSums[0];
    }
}

// 检查结果
bool check(float *out, float *res, int n) {
    const float epsilon = 0.01;
    for (int i = 0; i < n; i++) {
        if (fabs(out[i] - res[i]) > epsilon) return false;
    }
    return true;
}

void randomMatrix(float* matrix, int N) {
    for (int i = 0; i < N; i++) {
        matrix[i] = 2.0 * (float)drand48() - 1.0;
    }
}

template<int num_per_block, int block_num>
void cpuReduce(float* h_input, float* res) {
    for (int i = 0; i < block_num; i++) {
        float sum = 0;
        for (int j = 0; j < num_per_block; j++) {
            sum += h_input[i * num_per_block + j];
        }
        res[i] = sum;
    }
}

int main() {
    const int N = 32 * 1024 * 1024;
    const int block_num = 1024;
    const int num_per_block = N / block_num;  // 32768

    // 分配主机内存
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(block_num * sizeof(float));

    // 初始化输入数据
    randomMatrix(h_input, N);

    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, block_num * sizeof(float));

    // 数据拷贝到设备
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数
    dim3 grid(block_num);
    dim3 block(THREAD_PER_BLOCK);
    reduce<num_per_block><<<grid, block>>>(d_input, d_output);

    // 拷贝结果回主机
    cudaMemcpy(h_output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");
    // 计算正确结果
    float *res = (float*)malloc(block_num * sizeof(float));
    cpuReduce<num_per_block, block_num>(h_input, res);
    for (int i = 0; i < 10; i++) {
        printf("%f ", res[i]);
    }
    printf("\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_output[i]);
    }
    // 验证结果
    if (check(h_output, res, block_num)) {
        printf("结果正确！\n");
    } else {
        printf("结果错误！\n");
    }

    // 释放内存
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    free(res);

    return 0;
}