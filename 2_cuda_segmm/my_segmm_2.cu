#include<cuda_runtime.h>
#include<stdio.h>

template<int BLOCK, int STRIDE>
__global__ void segmm(float* d_A, float* d_B, float* d_C, int M, int K, int N) {
    const int STEP = STRIDE * BLOCK;
    __shared__ float d_A_shared[STEP][STEP];
    __shared__ float d_B_shared[STEP][STEP];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float* d_A_start = d_A + blockIdx.y * STEP * K;
    float* d_B_start = d_B + blockIdx.x * STEP;

    float sum[STRIDE][STRIDE] = {0.0f};
    for (int s = 0; s < K; s += STEP) {
        for (int i = 0; i < STRIDE; i++) {
            for (int j = 0; j < STRIDE; j++) {
                d_A_shared[ty * STRIDE + i][tx * STRIDE + j] = d_A_start[(ty * STRIDE + i) * K + s + tx * STRIDE + j];
                d_B_shared[ty * STRIDE + i][tx * STRIDE + j] = d_B_start[(s + ty * STRIDE + i) * N + tx * STRIDE + j];
            }
        } 
        __syncthreads();
        for (int i = 0; i < STRIDE; i++) {
            for (int j = 0; j < STRIDE; j++) {
                for (int k = 0; k < STEP; k++) {
                    sum[i][j] += d_A_shared[ty * STRIDE + i][k] * d_B_shared[k][tx * STRIDE + j];
                }
            }
        }
        __syncthreads();
    }
    int bx = blockIdx.x;
    int by = blockIdx.y;
    for (int i = 0; i < STRIDE; ++i) {
        for (int j = 0; j < STRIDE; ++j) {
            d_C[(STEP * by + ty * STRIDE + i) * N + STEP * bx + tx * STRIDE + j] = sum[i][j];
        }
    }
}

void randomMatrix(float* matrix, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            matrix[i * col + j] = 2.0 * (float)drand48() - 1.0;
        }
    }
}

//cpu的矩阵乘计算方法
void cpu_segmm(float* h_A, float* h_B, float* res, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            //每次读取h_A的第i行的第k个元素和h_B的第j列的第k个元素相乘
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            res[i * N + j] = sum;
        }
    }
}


bool check(float *out, float *res, int n) {
    const float epsilon = 0.01;
    for (int i = 0; i < n; i++) {
        if (fabs(out[i] - res[i]) > epsilon) return false;
    }
    return true;
}

int main() {
    const int m = 1024;
    const int k = 1024;
    const int n = 1024;
    
    //分配主机内存
    float* h_A, *h_B, *h_C;
    h_A = (float*)malloc(m * k * sizeof(float));
    h_B = (float*)malloc(k * n * sizeof(float));
    h_C = (float*)malloc(m * n * sizeof(float));

    //输入矩阵初始化
    randomMatrix(h_A, m, k);
    randomMatrix(h_B, k, n);

    //分配设备内存
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    //将主机复制到GPU上
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    //调用核函数
    dim3 block(16, 16);
    const int STRIDE = 2;
    dim3 grid((m + 16 - 1) / 16 / STRIDE, (n + 16 - 1) / 16 / STRIDE);
    segmm<16, STRIDE><<<grid, block>>>(d_A, d_B, d_C, m, k, n);

    //将结果拷贝回主机
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    float* res = (float*)malloc(m * n * sizeof(float));
    cpu_segmm(h_A, h_B, res, m, k, n);

    for (int i = 0; i < 10; i++) {
        printf("%f ", res[i]);
    }
    printf("\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_C[i]);
    }

    if (check(res, h_C, m * n)) {
        printf("ans is right\n");
    }
    else {
        printf("ans is error\n");
    }
    
    return 0;
}
