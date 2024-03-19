#include <iostream>
#include <vector>
#include <nvToolsExt.h>

#include <stdint.h>
#include <cuda_runtime_api.h>

#include "common.cuh"
#include "profile.cuh"
#include "kernel.cuh"

constexpr uint64_t row_size = 1 << 8;
constexpr uint64_t num_row = 1 << 8;
constexpr uint64_t elem_cnt = row_size * num_row;
constexpr double nnz_ratio = 0.5f;
constexpr uint64_t nnz = (uint64_t)((double)elem_cnt*nnz_ratio);
using data_type = float;

// we will place 32 threads on a row, so we will need 32 * num_row threads in total
constexpr int block_size = 1 << 8;
constexpr int grid_size =   32 * num_row % block_size == 0 ?
                            32 * num_row / block_size :
                            32 * num_row / block_size + 1;

int main(){
    PROFILE(nvtxRangePush("allocate host memory for vectors");)
    std::vector<uint64_t> col_ids;      // CSR col indices
    std::vector<uint64_t> row_ptr;      // CSR row pointers
    std::vector<float> data;            // CSR data
    std::vector<float> x;               // the multiplied vector
    x.reserve(row_size);
    std::vector<float> y;               // the result vector
    y.reserve(num_row);
    PROFILE(nvtxRangePop();)

    // initialize random value
    PROFILE(nvtxRangePush("initialize matrix and vector with random numbers");)
    generate_random_csr<data_type>(row_size, elem_cnt, nnz, col_ids, row_ptr, data);
    for (int i=0; i<row_size; i++){
        x.push_back(generate_random_value<data_type>());
    }
    PROFILE(nvtxRangePop();)
    
    // allocate memory space on device
    PROFILE(nvtxRangePush("allocate memory space on device");)
    uint64_t *d_col_ids, *d_row_ptr;
    float *d_data, *d_x, *d_y;
    GPU_ERROR(cudaMalloc(&d_col_ids,  sizeof(uint64_t)*col_ids.size()));
    GPU_ERROR(cudaMalloc(&d_row_ptr,  sizeof(uint64_t)*row_ptr.size()));
    GPU_ERROR(cudaMalloc(&d_data,     sizeof(data_type)*data.size()));
    GPU_ERROR(cudaMalloc(&d_x,        sizeof(data_type)*x.size()));
    GPU_ERROR(cudaMalloc(&d_y,        sizeof(data_type)*num_row));
    PROFILE(nvtxRangePop();)

    // copy data from host memory to device memory
    PROFILE(nvtxRangePush("copy data from host to device memory");)
    GPU_ERROR(cudaMemcpy(d_col_ids, col_ids.data(), sizeof(uint64_t)*col_ids.size(), cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(uint64_t)*row_ptr.size(), cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_data, data.data(), sizeof(data_type)*data.size(), cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_x, x.data(), sizeof(data_type)*x.size(), cudaMemcpyHostToDevice));
    PROFILE(nvtxRangePop();)

    // launch kernel
    PROFILE(
        std::cout << "Launch Kernel: " 
            << "block_size: " << block_size << ", " 
            << "grid_size: " << grid_size << std::endl;
        nvtxRangePush("launch kernel");
    )
    csr_spmv_kernel<data_type><<<grid_size, block_size>>>(num_row, d_col_ids, d_row_ptr, d_data, d_x, d_y);
    PROFILE(
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
        nvtxRangePop();
    )
    
    // copy result back to host memory
    PROFILE(nvtxRangePush("copy vector from device to host memory");)
    GPU_ERROR(cudaMemcpy(y.data(), d_y, sizeof(data_type)*num_row, cudaMemcpyDeviceToHost));
    PROFILE(nvtxRangePop();)

    // verify result
    verify_spmv_result<data_type>(num_row, col_ids, row_ptr, data, x, y);

    // free device memory
    PROFILE(nvtxRangePush("free device memory");)
    GPU_ERROR(cudaFree(d_col_ids));
    GPU_ERROR(cudaFree(d_row_ptr));
    GPU_ERROR(cudaFree(d_data));
    GPU_ERROR(cudaFree(d_x));
    GPU_ERROR(cudaFree(d_y));
    PROFILE(nvtxRangePop();)

    std::cout << "Get correct SpMV result!" << std::endl;
}
