#include<cuda_runtime.h>
#include<stdio.h>
#include<algorithm>
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])



__global__ void mySoftMax(float* d_x, float* d_y, int N, float* sum, float* max_ele) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    d_y[index] = std::exp(d_x[index] - *max_ele) / (*sum); 
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

void cpuSoftMax(float* h_x, float* res, int N){
    int max_ele = *(std::max_element(h_x, h_x + N));
    //求和
    float sum = 0.f;
    for (int i = 0; i < N; i++) {
        res[i] = std::exp(h_x[i] - max_ele);
        sum += res[i];
    }
    for (int i = 0; i < N; i++) {
        res[i] = res[i] / sum;
    }
}

int main() {
    const int n = 1024 * 16;
    
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
    mySoftMax<<<grid, block>>>(d_x, d_y, n);

    //将结果拷贝回主机
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    float* res = (float*)malloc(n * sizeof(float));
    cpuSoftMax(h_x, res, n);
    if (check(res, h_y, n)) {
        printf("ans is right\n");
    }
    else {
        printf("ans is error\n");
    }
    
    return 0;
}
