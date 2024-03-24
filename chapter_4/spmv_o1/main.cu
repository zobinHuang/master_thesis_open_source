#include <iostream>
#include <sstream>
#include <vector>
#include <nvToolsExt.h>

#include <stdint.h>
#include <unistd.h>
#include <cuda_runtime_api.h>

#include "profile.cuh"
#include "utils.cuh"
#include "kernel.cuh"

uint64_t kRowSize           = 1 << 10;
uint64_t kNumRow            = 1 << 16;
uint64_t kElemCnt = kRowSize * kNumRow;
double kNnzRatio = 0.5f;
double kNnzDistribution     = static_cast<double>(1 << 5);
uint64_t kNnzCnt = (uint64_t)((double)kElemCnt*kNnzRatio);
using kDataType = float;
using kIndexType = uint64_t;

// we will place 32 threads on a row, so we will need 32 * kNumRow threads in total
int kBlockSize = 1 << 8;
int kGridSize =   32 * kNumRow % kBlockSize == 0 ?
                    32 * kNumRow / kBlockSize :
                    32 * kNumRow / kBlockSize + 1;

template<typename T>
T __string_to_num(std::string a){
    std::istringstream iss(a);
    T num;
    iss >> num;
    return num;
}

void __parse_cli(int argc, char** argv){
    char ch;
    const char* optstr = "m:n:r:d:";
    while( (ch = getopt(argc, argv, optstr)) != -1 ){
        switch(ch)
        {
        case 'm':
            kRowSize = atoll(optarg);
            break;
        case 'n':
            kNumRow = atoll(optarg);
            break;
        case 'r':
            kNnzRatio = __string_to_num<double>(optarg);
            break;
        case 'd':
            kNnzDistribution = __string_to_num<double>(optarg);
            break;
        default:
            assert(0);
        }
    }

    kElemCnt = kRowSize * kNumRow;
    kNnzCnt = (uint64_t)((double)kElemCnt*kNnzRatio);
    kGridSize =     32 * kNumRow % kBlockSize == 0 ?
                    32 * kNumRow / kBlockSize :
                    32 * kNumRow / kBlockSize + 1;
}

int main(int argc, char** argv){
    __parse_cli(argc, argv);
    std::cout
        << "m=" << std::to_string(kRowSize) 
        << ", n=" << std::to_string(kNumRow) 
        << ", ratio=" << std::to_string(kNnzRatio)  
        << ", distribute=" << std::to_string(kNnzDistribution)
        << std::endl;
    
    PROFILE(nvtxRangePush("allocate host memory for vectors");)
    std::vector<uint64_t> col_ids;      // CSR col indices
    std::vector<uint64_t> row_ptr;      // CSR row pointers
    std::vector<float> data;            // CSR data
    std::vector<float> x;               // the multiplied vector
    x.reserve(kRowSize);
    std::vector<float> y;               // the result vector
    y.reserve(kNumRow);
    PROFILE(nvtxRangePop();)

    // initialize random value
    PROFILE(nvtxRangePush("initialize matrix and vector with random numbers");)
    generate_random_csr<kDataType, kIndexType>(kRowSize, kElemCnt, kNnzCnt, col_ids, row_ptr, data, kNnzDistribution);
    for (int i=0; i<kRowSize; i++){
        x.push_back(generate_random_value<kDataType>());
    }
    PROFILE(nvtxRangePop();)
    
    // allocate memory space on device
    PROFILE(nvtxRangePush("allocate memory space on device");)
    uint64_t *d_col_ids, *d_row_ptr;
    float *d_data, *d_x, *d_y;
    GPU_ERROR(cudaMalloc(&d_col_ids,  sizeof(uint64_t)*col_ids.size()));
    GPU_ERROR(cudaMalloc(&d_row_ptr,  sizeof(uint64_t)*row_ptr.size()));
    GPU_ERROR(cudaMalloc(&d_data,     sizeof(kDataType)*data.size()));
    GPU_ERROR(cudaMalloc(&d_x,        sizeof(kDataType)*x.size()));
    GPU_ERROR(cudaMalloc(&d_y,        sizeof(kDataType)*kNumRow));
    PROFILE(nvtxRangePop();)

    // copy data from host memory to device memory
    PROFILE(nvtxRangePush("copy data from host to device memory");)
    GPU_ERROR(cudaMemcpy(d_col_ids, col_ids.data(), sizeof(kIndexType)*col_ids.size(), cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(kIndexType)*row_ptr.size(), cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_data, data.data(), sizeof(kDataType)*data.size(), cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_x, x.data(), sizeof(kDataType)*x.size(), cudaMemcpyHostToDevice));
    PROFILE(nvtxRangePop();)

    // launch kernel
    PROFILE(
        std::cout << "Launch Kernel: " 
            << "kBlockSize: " << kBlockSize << ", " 
            << "kGridSize: " << kGridSize << std::endl;
        nvtxRangePush("launch kernel");
    )
    csr_spmv_kernel<kDataType, kIndexType><<<kGridSize, kBlockSize>>>(kNumRow, d_col_ids, d_row_ptr, d_data, d_x, d_y);
    PROFILE(
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
        nvtxRangePop();
    )
    
    // copy result back to host memory
    PROFILE(nvtxRangePush("copy vector from device to host memory");)
    GPU_ERROR(cudaMemcpy(y.data(), d_y, sizeof(kDataType)*kNumRow, cudaMemcpyDeviceToHost));
    PROFILE(nvtxRangePop();)

    // verify result
    verify_spmv_result<kDataType>(kNumRow, col_ids, row_ptr, data, x, y);

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
