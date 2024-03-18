#include <iostream>
#include <vector>

#include <stdint.h>
#include <cuda_runtime_api.h>

/*!
 *  \brief  naive implementation of SpMV kernel
 *  \param  n_rows  number of rows within the source matrix
 *  \param  col_idx CSR column indices
 *  \param  row_ptr CSR row pointers
 *  \param  value   CSR values
 *  \param  vector  source vector
 *  \param  result  result vector
 */
template <typename T>
__global__ void csr_spmv_kernel (
    const uint64_t n_rows,
    const uint64_t *col_idx,
    const uint64_t *row_ptr,
    const T *value,
    const T *vector,
    T *result
){
    // matrix row to be processed
    uint32_t row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(row_id < n_rows){
        const int row_start = row_ptr[row_id];
        const int row_end = row_ptr[row_id+1];
        T sum = 0;
        for(uint32_t i = row_start; i < row_end; i++){
            sum += value[i] * vector[col_idx[i]];
        }
        result[row_id] = sum;
    }
}
