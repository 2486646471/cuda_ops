#include<cuda_runtime.h>
#include<stdio.h>
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void myRelu(float* d_x, float* d_y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 tmp_x = FLOAT4(d_x[idx]);
        float4 tmp_y;
        tmp_y.x = fmax(0.f, tmp_x.x);
        tmp_y.y = fmax(0.f, tmp_x.y);
        tmp_y.z = fmax(0.f, tmp_x.z);
        tmp_y.w = fmax(0.f, tmp_x.w);
        FLOAT4(d_y[idx]) = tmp_y;
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

void cpuRelu(float* h_x, float* res, int N){
    for (int i = 0; i < N; i++) {
        res[i] = fmax(0.f, h_x[i]);
    }
}

int main() {
    const int n = 1024 * 1024;
    
    //分配主机内存
    float* h_x, *h_y;
    h_x = (float*)malloc(n * sizeof(float));
    h_y = (float*)malloc(n * sizeof(float));

    //输入矩阵初始化
    randomMatrix(h_x, n);
    randomMatrix(h_y, n);

    //分配设备内存
    float* d_x, *d_y;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    //将主机复制到GPU上
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
  
    //调用核函数
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    myRelu<<<grid, block>>>(d_x, d_y, n);

    //将结果拷贝回主机
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    float* res = (float*)malloc(n * sizeof(float));
    cpuRelu(h_x, res, n);
    for (int i = 0; i < 10; i++) {
        printf("%f ", res[i]);
    }
    printf("\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_y[i]);
    }
    if (check(res, h_y, n)) {
        printf("ans is right\n");
    }
    else {
        printf("ans is error\n");
    }
    
    return 0;
}
