#include <iostream>
#include <vector>

#include <stdint.h>
#include <cuda_runtime_api.h>

/*!
 *  \brief  obtain the latest power of 2 value before given value
 *  \param  n   given value
 *  \return the latest power of 2 value before given value
 */
template<typename IndexType>
__device__ IndexType __prev_power_of_2(IndexType n) {
    while (n & n - 1) { n = n & n - 1; }
    return n;
}

#define FULL_WARP_MASK 0xffffffff

/*!
 *  \brief  conduct reduction adding within the warp
 *  \param  val the value to be reduced
 *  \return the reduction result
 *          (only the result from first thread of the warp is meanful)
 */
template <typename T>
__device__ T __warp_reduce(T val){
    int i;
    for(i = 32 / 2; i > 0; i /= 2){
        val += __shfl_down_sync(FULL_WARP_MASK, val, i);
    }
    return val;
}


/*!
 *  \brief  stream implementation of SpMV kernel
 *  \param  row_ptr_begin_id    begin index within row_ptr
 *  \param  row_ptr_end_id      end index within row_ptr
 *  \param  col_idx_begin_id    begin index within col_idx
 *  \param  col_idx_end_id      end index within col_idx
 *  \param  block_nnz           number of nnz that this block processes
 *  \param  row_ptr             CSR row pointers
 *  \param  col_idx             CSR column indices
 *  \param  value               CSR values
 *  \param  vector              source vector
 *  \param  result              result vector
 *  \param  cache               shared memory cache to store the values to be reduced
 */
template <typename T, typename IndexType, int kBlockSize>
__device__ void __csr_spmv_stream(
    const IndexType row_ptr_begin_id,
    const IndexType row_ptr_end_id,
    const IndexType col_idx_begin_id,
    const IndexType col_idx_end_id,
    const IndexType block_nnz,
    const IndexType* row_ptr,
    const IndexType* col_idx,
    const T* value,
    const T* vector,
    T* result,
    T* cache
){
    /* ========== step 1: map  ========== */
    const IndexType col_idx_thread_id = col_idx_begin_id + threadIdx.x;
    if (threadIdx.x < block_nnz) {
        cache[threadIdx.x] = value[col_idx_thread_id] * vector[col_idx[col_idx_thread_id]];
    }
    __syncthreads();

    /* ========== step 2: reduce ========== */
    const IndexType block_num_rows = row_ptr_end_id - row_ptr_begin_id;
    const IndexType reduction_group_size = __prev_power_of_2<IndexType>(kBlockSize / block_num_rows);

    if (reduction_group_size > 1) {
        /*!
         *  \note   case:   reduce all non zeroes of the row by multiple threads,
         *                  imply that the number of processing rows is lees than kBlockSize
         */
        const IndexType reduction_group_id = threadIdx.x / reduction_group_size;
        const IndexType target_row_id = row_ptr_begin_id + reduction_group_id;
        const IndexType thread_id_in_reduction_group = threadIdx.x - reduction_group_id * reduction_group_size;
        
        T reduction_result = 0.0;

        if (target_row_id < row_ptr_end_id) {
            /*!
             *  \note   obtain the indices inside shared memory cache for reduction of current thread
                        actually it's the logic index of nnz
             */
            const IndexType cache_begin_id = row_ptr[target_row_id] - row_ptr[row_ptr_begin_id];
            const IndexType cache_end_id = row_ptr[target_row_id + 1] - row_ptr[row_ptr_begin_id];

            for (IndexType j = cache_begin_id + thread_id_in_reduction_group; j < cache_end_id; j += reduction_group_size) {
                reduction_result += cache[j];
            }
        }

        // wait all threads within the reduction group to finish merging
        __syncthreads();

        // write back the reduction group result to shared memory cache
        cache[threadIdx.x] = reduction_result;

        // final reduce of the reduction group
        for (IndexType j = reduction_group_size / 2; j > 0; j /= 2) {
            __syncthreads();
            const bool use_result = (thread_id_in_reduction_group < j) && (threadIdx.x + j < kBlockSize);
            if (use_result) { reduction_result += cache[threadIdx.x + j]; }
            __syncthreads();
            if (use_result) { cache[threadIdx.x] = reduction_result; }
        }

        // write back result
        if (thread_id_in_reduction_group == 0 && target_row_id < row_ptr_end_id) {
            result[target_row_id] = reduction_result;
        }
    } else {
        /*!
         *  \note   case:   reduce all non zeroes of row by a single thread,
         *                  imply that the number of processing rows is larger or equal to kBlockSize
         */
        IndexType target_row_id = row_ptr_begin_id + threadIdx.x;

        while (target_row_id < row_ptr_end_id) {
            /*!
             *  \note   obtain the indices inside shared memory cache for reduction of current thread
                        actually it's the logic index of nnz
             */
            const IndexType cache_begin_id = row_ptr[target_row_id] - row_ptr[row_ptr_begin_id];
            const IndexType cache_end_id = row_ptr[target_row_id + 1] - row_ptr[row_ptr_begin_id];

            // initialize reduction result of current thread
            T reduction_result = 0.0;

            // reduction loop
            for (IndexType j = cache_begin_id; j < cache_end_id; j++) { reduction_result += cache[j]; }

            // write back result
            result[target_row_id] = reduction_result;

            // jump across kBlockSize
            // prevent the case that the number of merged rows is larger than kBlockSize
            target_row_id += kBlockSize;
        }
    }
}


/*!
 *  \brief  vector implementation of SpMV kernel
 *  \param  num_rows            number of rows within the source matrix
 *  \param  target_row_id       target processing row id
 *  \param  row_ptr_begin_id    begin index within row_ptr
 *  \param  row_ptr_end_id      end index within row_ptr
 *  \param  col_idx_begin_id    begin index within col_idx
 *  \param  col_idx_end_id      end index within col_idx
 *  \param  block_nnz           number of nnz that this block processes
 *  \param  row_ptr             CSR row pointers
 *  \param  col_idx             CSR column indices
 *  \param  value               CSR values
 *  \param  vector              source vector
 *  \param  result              result vector
 *  \param  cache               shared memory cache to store the values to be reduced
 */
template<typename T, typename IndexType, int kBlockSize>
__device__ void __csr_spmv_vector_l(
    const IndexType num_rows,
    const IndexType target_row_id,
    const IndexType col_idx_begin_id,
    const IndexType col_idx_end_id,
    const IndexType block_nnz,
    const IndexType* row_ptr,
    const IndexType* col_idx,
    const T* value,
    const T* vector,
    T* result,
    T* cache
){
    const IndexType warp_id = threadIdx.x / 32;
    const IndexType lane_id = threadIdx.x - warp_id * 32;

    T reduction_result = 0.0;

    if (target_row_id < num_rows) {  // might no need to judge here
        // reduction loop
        for (IndexType j = col_idx_begin_id + threadIdx.x; j < col_idx_end_id; j += kBlockSize) {
            reduction_result += value[j] * vector[col_idx[j]];
        }
    }

    // use warp primitive for reduction within the warp
    reduction_result = __warp_reduce<T>(reduction_result);

    // store the reduction result of current warp to shared memory cache
    if (lane_id == 0) { cache[warp_id] = reduction_result; }
    __syncthreads();

    // use the first warp to reduce final result
    if (warp_id == 0) {
        reduction_result = 0.0;
        for (IndexType j = lane_id; j < kBlockSize / 32; j += 32) { reduction_result += cache[j]; }
        reduction_result = __warp_reduce<T>(reduction_result);

        // use the first lane to write back result
        if (lane_id == 0 && target_row_id < num_rows) {  // might no need the last judgement here
            result[target_row_id] = reduction_result;
        }
    }
}


/*!
 *  \brief  vector implementation of SpMV kernel
 *  \param  num_rows            number of rows within the source matrix
 *  \param  target_row_id       target processing row id
 *  \param  col_idx_begin_id    begin index within col_idx
 *  \param  col_idx_end_id      end index within col_idx
 *  \param  block_nnz           number of nnz that this block processes
 *  \param  row_ptr             CSR row pointers
 *  \param  col_idx             CSR column indices
 *  \param  value               CSR values
 *  \param  vector              source vector
 *  \param  result              result vector
 *  \param  cache               shared memory cache to store the values to be reduced
 */
template <typename T, typename IndexType>
__device__ void __csr_spmv_vector (
    const IndexType num_rows,
    const IndexType target_row_id,
    const IndexType col_idx_begin_id,
    const IndexType col_idx_end_id,
    const IndexType block_nnz,
    const IndexType* row_ptr,
    const IndexType* col_idx,
    const T* value,
    const T* vector,
    T* result
){
    const IndexType warp_id = threadIdx.x / 32;
    const IndexType lane_id = threadIdx.x - warp_id * 32;

    T reduction_result = 0.0;

    if (target_row_id < num_rows) {  // might no need to judge here
        // reduction loop
        for (IndexType j = col_idx_begin_id + lane_id; j < col_idx_end_id; j += 32) {
            reduction_result += value[j] * vector[col_idx[j]];
        }
    }

    // use warp primitive for reduction within the warp
    reduction_result = __warp_reduce<T>(reduction_result);

    if (lane_id == 0 && warp_id == 0 && target_row_id < num_rows) {  // might no need the last judgement here
        result[target_row_id] = reduction_result;
    }
}



/*!
 *  \brief  naive implementation of SpMV kernel
 *  \param  n_rows  number of rows within the source matrix
 *  \param  col_idx CSR column indices
 *  \param  row_ptr CSR row pointers
 *  \param  value   CSR values
 *  \param  vector  source vector
 *  \param  result  result vector
 */
template <typename T, typename IndexType, int kBlockSize>
__global__ void csr_spmv_kernel (
    const IndexType n_rows,
    const IndexType* row_assignment,
    const IndexType* col_idx,
    const IndexType* row_ptr,
    const T* value,
    const T* vector,
    T *result
){
    // start index of row_ptr
    const IndexType row_ptrs_begin_id = row_assignment[blockIdx.x];

    // end index of row_ptr (not included)
    const IndexType row_ptrs_end_id = row_assignment[blockIdx.x + 1];

    // obtain range of element indices for current block
    // start index of col_ids
    const IndexType col_idx_begin_id = row_ptr[row_ptrs_begin_id];  

    // end index of col_ids (not included)
    const IndexType col_idx_end_id = row_ptr[row_ptrs_end_id];

    // obtain number of nnz to be processed for current block
    const IndexType block_nnz = col_idx_end_id - col_idx_begin_id;

    // declare shared memory for synchronization among threads in block
    __shared__ T cache[kBlockSize];

    // invoke different rotinue according to nnz distribution
    if (row_ptrs_end_id - row_ptrs_begin_id > 1) {
        // case: more than one row be assigned to current block
        __csr_spmv_stream<T, IndexType, kBlockSize>(
            row_ptrs_begin_id, row_ptrs_end_id, col_idx_begin_id,
            col_idx_end_id, block_nnz, row_ptr, col_idx, value, vector,
            result, cache
        );
    } else {
        // case: only one row be assigned to current block
        if (block_nnz <= 64 || kBlockSize <= 32) {
            // all warps within __csr_spmv_vector are doing the exact same thing (duplicated),
            // so we don't want too many threads in it
            __csr_spmv_vector<T, IndexType>(
                n_rows, row_ptrs_begin_id, col_idx_begin_id, col_idx_end_id,
                block_nnz, row_ptr, col_idx, value, vector, result
            );
        } else {
            __csr_spmv_vector_l<T, IndexType, kBlockSize>(
                n_rows, row_ptrs_begin_id, col_idx_begin_id, col_idx_end_id,
                block_nnz, row_ptr, col_idx, value, vector, result, cache
            );
        }
    }
}

/*!
 *  \brief  assign different rows within the matrix to different SpMV algorithm
 *  \param  row_ptr CSR row pointers
 *  \param  return  assignment result
 */
template <typename T, typename IndexType, int kBlockSize>
std::vector<IndexType> infer_row_assignment(const std::vector<IndexType> &row_ptr){
    IndexType last_row_id = 0, nnz_sum = 0, i;
    std::vector<IndexType> row_assignment;

#define UPDATE_ASSIGNMENT_META(row_id)          \
            row_assignment.push_back(row_id);   \
            last_row_id = row_id;               \
            nnz_sum = 0;

    row_assignment.push_back(0);
    
    for(i=1; i <= row_ptr.size(); i++){
        nnz_sum += row_ptr[i] - row_ptr[i-1];
        if (nnz_sum == kBlockSize) {
            /*! 
             *  \note   case:   number of scanned nnz equals to kBlockSize
             *                  then we can close this assignment
             */
            UPDATE_ASSIGNMENT_META(i);
        } else if(nnz_sum > kBlockSize){
            /*! 
             *  \note   case:   number of scanned nnz exceeds kBlockSize
             *                  then we would abandon current row if we already
             *                  have more than one row, and close this assignment
             */
            if (i - last_row_id > 1) { i--; }
            UPDATE_ASSIGNMENT_META(i);
        } else if (i - last_row_id > kBlockSize) {
            /*! 
             *  \note   case (corner):  number of merged rows exceeds kBlockSize
             *                          then we also need to close this assignment
             */
            UPDATE_ASSIGNMENT_META(i);
        }
    }

#undef UPDATE_ASSIGNMENT_META

    row_assignment.push_back(static_cast<IndexType>(row_ptr.size()));
    return row_assignment;
}
