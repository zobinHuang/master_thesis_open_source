#include <iostream>
#include <vector>

#include <stdint.h>
#include <cuda_runtime_api.h>

#define FULL_WARP_MASK 0xffffffff

/*!
 *  \brief  conduct reduction adding within the warp
 *  \param  val the value to be reduced
 *  \return the reduction result
 *          (only the result from first thread of the warp is meanful)
 */
template <typename T>
__device__ T __warp_reduce(T val){
    uint64_t i;
    for(i = 32 / 2; i > 0; i /= 2){
        val += __shfl_down_sync(FULL_WARP_MASK, val, i);
    }
    return val;
}

/*!
 *  \brief  per-warp version of SpMV kernel
 *  \param  n_rows  number of rows within the source matrix
 *  \param  col_idx CSR column indices
 *  \param  row_ptr CSR row pointers
 *  \param  value   CSR values
 *  \param  vector  source vector
 *  \param  result  result vector
 */
template <typename T, typename kIndexType>
__global__ void csr_spmv_kernel (
    const kIndexType n_rows,
    const kIndexType *col_idx,
    const kIndexType *row_ptr,
    const T *value,
    const T *vector,
    T *result
){
    const kIndexType thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const kIndexType warp_id = thread_id / 32;
    const kIndexType lane_id = thread_id - warp_id * 32;
    const kIndexType row_id = warp_id;

    T sum = 0;

    if (row_id < n_rows) {
        const kIndexType row_start = row_ptr[row_id];
        const kIndexType row_end = row_ptr[row_id+1];
        
        for(kIndexType i=row_start+lane_id; i<row_end; i+=32){
            sum += value[i] * vector[col_idx[i]];
        }
    }

    // sync within the warp
    sum = __warp_reduce<T>(sum);

    if(lane_id == 0 && row_id < n_rows){
        result[row_id] = sum;
    }
}
