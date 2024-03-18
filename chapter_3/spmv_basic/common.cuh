#pragma once

#include <iostream>
#include <vector>

#include <stdint.h>
#include <cuda_runtime_api.h>

template<typename T>
T generate_random_value();

template<typename T>
void csr_spmv_cpu_kernel(
    const uint64_t n_rows,
    const std::vector<uint64_t> &col_idx,
    const std::vector<uint64_t> &row_ptr,
    const std::vector<T> &value,
    const std::vector<T> &vector,
    std::vector<T> &result
);


template<typename T>
void verify_spmv_result(
    const uint64_t n_rows,
    const std::vector<uint64_t> &col_idx,
    const std::vector<uint64_t> &row_ptr,
    const std::vector<T> &value,
    const std::vector<T> &vector,
    const std::vector<T> &result
);


template<typename T>
void generate_random_csr(
    const uint64_t row_size,
    const uint64_t elem_cnt,    
    const uint64_t nnz,
    std::vector<uint64_t> &col_ids,
    std::vector<uint64_t> &row_ptr,
    std::vector<T> &data
);



inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPUassert: \"" << cudaGetErrorString(code) << "\"  in "
              << file << ": " << line << "\n";
    if (abort)
      exit(code);
  }
}
#define GPU_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
