#include "time.hpp"
#include "gpu_error.h"

#include <unistd.h>
#include <iostream>
#include <nvml.h>

__global__ void powerKernel(double* A, int iters) {
    int tidx = threadIdx.x + blockIdx.x*blockDim.x;

    double start = A[0];
    #pragma unroll 1
    for(int i = 0; i < iters; i++) {
        start -= (tidx*0.1)*start;
    }
    A[0] = start;
}

unsigned int getGPUClock() {

    double* dA = NULL;
    unsigned int gpu_clock;
    int iters = 10;

    GPU_ERROR(cudaMalloc(&dA, sizeof(double)));

    powerKernel<<<1000, 1024>>>(dA, iters);

    double dt = 0;
    std::cout << "clock: ";
    while (dt < 0.4) {
        GPU_ERROR(cudaDeviceSynchronize());
        double t1 = get_time_s();
        powerKernel<<<1000, 1024>>>(dA, iters);
        usleep(10000);

        nvmlInit();
        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(0, &device);
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &gpu_clock);
        GPU_ERROR(cudaDeviceSynchronize());
        double t2 = get_time_s();
        std::cout << gpu_clock << " ";
        std::cout.flush();
        dt = t2 - t1;
        iters *= 2;
    }
    std::cout << "\n";

    GPU_ERROR(cudaFree(dA));

    return gpu_clock;
}
