#include<cuda_runtime.h>
#include<stdio.h>
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void myAdd(float* d_A, float* d_B, float* d_C, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 tmp_a = FLOAT4(d_A[idx]);
        float4 tmp_b = FLOAT4(d_B[idx]);
        float4 tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        tmp_c.z = tmp_a.z + tmp_b.z;
        tmp_c.w = tmp_a.w + tmp_b.w;
        FLOAT4(d_C[idx]) = tmp_c;
    }
}

void randomMatrix(float* matrix, int N) {
    for (int i = 0; i < N; i++) {
        matrix[i] = 2.0 * (float)drand48() - 1.0;
    }
}

bool check(float *out, float *res, int N) {
    const float epsilon = 0.01;
    for (int i = 0; i < N; i++) {
        if (fabs(out[i] - res[i]) > epsilon) return false;
    }
    return true;
}

void cpuAdd(float* h_A, float* h_B, float* res, int N){
    for (int i = 0; i < N; i++) {
        res[i] = h_A[i] + h_B[i];
    }
}

int main() {
    const int n = 1024 * 1024;
    
    //分配主机内存
    float* h_A, *h_B, *h_C;
    h_A = (float*)malloc(n * sizeof(float));
    h_B = (float*)malloc(n * sizeof(float));
    h_C = (float*)malloc(n * sizeof(float));

    //输入矩阵初始化
    randomMatrix(h_A, n);
    randomMatrix(h_B, n);

    //分配设备内存
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n * sizeof(float));
    cudaMalloc((void**)&d_B, n * sizeof(float));
    cudaMalloc((void**)&d_C, n * sizeof(float));

    //将主机复制到GPU上
    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice);

    //调用核函数
    dim3 block(256);
    dim3 grid(n + 255 / 256);
    myAdd<<<grid, block>>>(d_A, d_B, d_C, n);

    //将结果拷贝回主机
    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    float* res = (float*)malloc(n * sizeof(float));
    cpuAdd(h_A, h_B, res, n);
    for (int i = 0; i < 10; i++) {
        printf("%f ", res[i]);
    }
    printf("\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_C[i]);
    }
    if (check(res, h_C, n)) {
        printf("ans is right\n");
    }
    else {
        printf("ans is error\n");
    }
    
    return 0;
}
