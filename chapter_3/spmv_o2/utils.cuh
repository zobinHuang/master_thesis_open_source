#include <iostream>
#include <vector>
#include <cassert>
#include <stdint.h>
#include <sys/time.h>

#include <cuda_runtime_api.h>

/*!
 *  \brief  generate random value
 */
template<typename T>
T generate_random_value(){
    if( 
        typeid(T) == typeid(short) or 
        typeid(T) == typeid(int) or 
        typeid(T) == typeid(long) or 
        typeid(T) == typeid(unsigned short) or
        typeid(T) == typeid(unsigned int) or
        typeid(T) == typeid(unsigned long)
    ) {
        return rand()%100;
    } else if (
        typeid(T) == typeid(float) or 
        typeid(T) == typeid(double)
    ) {
        return static_cast<T>(rand())/static_cast<T>(RAND_MAX);
    } else {
        std::cout << "Unknown type name " << typeid(T).name() << std::endl;
        return static_cast<T>(1);
    }
}
 

/*!
 *  \brief  naive implementation of SpMV kernel (cpu)
 *  \param  n_rows  number of rows within the source matrix
 *  \param  col_idx CSR column indices
 *  \param  row_ptr CSR row pointers
 *  \param  value   CSR values
 *  \param  vector  source vector
 *  \param  result  result vector
 */
template<typename T>
void csr_spmv_cpu_kernel(
    const uint64_t n_rows,
    const std::vector<uint64_t> &col_idx,
    const std::vector<uint64_t> &row_ptr,
    const std::vector<T> &value,
    const std::vector<T> &vector,
    std::vector<T> &result
){
    for(uint64_t row=0; row<n_rows; row++){
        const uint64_t row_start = row_ptr[row];
        const uint64_t row_end = row_ptr[row+1];
        float sum = 0;
        for(uint64_t i=row_start; i<row_end; i++){
            sum += value[i] * vector[col_idx[i]];
        }
        result[row] = sum;
    }
}


/*!
 *  \brief  verify the computation result of SpMV kernel
 *  \param  n_rows  number of rows within the source matrix
 *  \param  col_idx CSR column indices
 *  \param  row_ptr CSR row pointers
 *  \param  value   CSR values
 *  \param  vector  source vector
 *  \param  result  result vector
 */
template<typename T, typename IndexType>
void verify_spmv_result(
    const IndexType n_rows,
    const std::vector<IndexType> &col_idx,
    const std::vector<IndexType> &row_ptr,
    const std::vector<T> &value,
    const std::vector<T> &vector,
    const std::vector<T> &result
){
    std::vector<T> correct_result(n_rows, 0);
    
    for(IndexType row=0; row<n_rows; row++){
        const IndexType row_start = row_ptr[row];
        const IndexType row_end = row_ptr[row+1];
        float sum = 0;
        for(IndexType i=row_start; i<row_end; i++){
            sum += value[i] * vector[col_idx[i]];
        }
        correct_result[row] = sum;
    }

    for(IndexType row=0; row<n_rows; row++){
        // std::cout << "result: " << result[row] << "; correct result: " << correct_result[row] << std::endl;
        assert(abs(result[row]-correct_result[row]) <= 0.00001);
    }
}


/*!
 * \brief generate random sparse matrix in CSR format
 * \param row_size  number of elements per row
 * \param elem_cnt  number of elements in the matrix
 * \param nnz       number of non-zero elements
 * \param col_ids   CSR column indices
 * \param row_ptr   CSR row pointers
 * \param data      CSR data
 */
template<typename T, typename IndexType>
void generate_random_csr(
    const IndexType row_size,
    const IndexType elem_cnt,    
    const IndexType nnz,
    std::vector<IndexType> &col_ids,
    std::vector<IndexType> &row_ptr,
    std::vector<T> &data
){
    // generate random sparse matrix
    std::vector<T> temp_array(elem_cnt, 0);
    for (IndexType i=0; i<nnz; i++) {
        IndexType index = (IndexType) (elem_cnt * ((double) rand() / (RAND_MAX + 1.0)));
        temp_array[index] = generate_random_value<T>();
    }

    // assert elem_cnt is divided by row_size
    assert(elem_cnt%row_size == 0);

    // convert to CSR format
    IndexType n_rows = elem_cnt/row_size;
    IndexType nnz_count = 0;
    row_ptr.push_back(0);
    for(IndexType row=0; row<n_rows; row++){
        IndexType nnz_row = 0;
        for(IndexType col=0; col<row_size; col++){
            if(temp_array[row*row_size+col] != 0){
                nnz_row += 1;
                col_ids.push_back(col);
                data.push_back(temp_array[row*row_size+col]);
            }
        }
        nnz_count += nnz_row;
        row_ptr.push_back(nnz_count);
    }
}

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
