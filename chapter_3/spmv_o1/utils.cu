#include <iostream>
#include <vector>
#include <cassert>
#include <stdint.h>
#include <sys/time.h>

#include "common.cuh"

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
#define FORCE_COMPILE_GENERATE_RANDOM_VALUE(type)   \
        template type generate_random_value<type>();    

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
#define FORCE_COMPILE_CSR_SPMV_CPU_KERNEL(type)     \
    template void csr_spmv_cpu_kernel<type>(        \
        const uint64_t n_rows,                      \
        const std::vector<uint64_t> &col_idx,       \
        const std::vector<uint64_t> &row_ptr,       \
        const std::vector<type> &value,             \
        const std::vector<type> &vector,            \
        std::vector<type> &result                   \
    );


/*!
 *  \brief  verify the computation result of SpMV kernel
 *  \param  n_rows  number of rows within the source matrix
 *  \param  col_idx CSR column indices
 *  \param  row_ptr CSR row pointers
 *  \param  value   CSR values
 *  \param  vector  source vector
 *  \param  result  result vector
 */
template<typename T>
void verify_spmv_result(
    const uint64_t n_rows,
    const std::vector<uint64_t> &col_idx,
    const std::vector<uint64_t> &row_ptr,
    const std::vector<T> &value,
    const std::vector<T> &vector,
    const std::vector<T> &result
){
    std::vector<T> correct_result(n_rows, 0);
    
    for(uint64_t row=0; row<n_rows; row++){
        const uint64_t row_start = row_ptr[row];
        const uint64_t row_end = row_ptr[row+1];
        float sum = 0;
        for(uint64_t i=row_start; i<row_end; i++){
            sum += value[i] * vector[col_idx[i]];
        }
        correct_result[row] = sum;
    }

    for(uint64_t row=0; row<n_rows; row++){
        // std::cout << "result: " << result[row] << "; correct result: " << correct_result[row] << std::endl;
        assert(abs(result[row]-correct_result[row]) <= 0.00001);
    }
}
#define FORCE_COMPILE_VERIFY_SPMV_RESULT(type)      \
    template void verify_spmv_result<type>(         \
        const uint64_t n_rows,                      \
        const std::vector<uint64_t> &col_idx,       \
        const std::vector<uint64_t> &row_ptr,       \
        const std::vector<type> &value,             \
        const std::vector<type> &vector,            \
        const std::vector<type> &result             \
    );

/*!
 * \brief generate random sparse matrix in CSR format
 * \param row_size  number of elements per row
 * \param elem_cnt  number of elements in the matrix
 * \param nnz       number of non-zero elements
 * \param col_ids   CSR column indices
 * \param row_ptr   CSR row pointers
 * \param data      CSR data
 */
template<typename T>
void generate_random_csr(
    const uint64_t row_size,
    const uint64_t elem_cnt,    
    const uint64_t nnz,
    std::vector<uint64_t> &col_ids,
    std::vector<uint64_t> &row_ptr,
    std::vector<T> &data
){
    // generate random sparse matrix
    std::vector<T> temp_array(elem_cnt, 0);
    for (uint64_t i=0; i<nnz; i++) {
        uint64_t index = (uint64_t) (elem_cnt * ((double) rand() / (RAND_MAX + 1.0)));
        temp_array[index] = generate_random_value<T>();
    }

    // assert elem_cnt is divided by row_size
    assert(elem_cnt%row_size == 0);

    // convert to CSR format
    uint64_t n_rows = elem_cnt/row_size;
    uint64_t nnz_count = 0;
    row_ptr.push_back(0);
    for(uint64_t row=0; row<n_rows; row++){
        uint64_t nnz_row = 0;
        for(uint64_t col=0; col<row_size; col++){
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
#define FORCE_COMPILE_GENERATE_RANDOM_CSR(type) \
    template void generate_random_csr<type>(    \
        const uint64_t row_size,                \
        const uint64_t elem_cnt,                \
        const uint64_t nnz,                     \
        std::vector<uint64_t> &col_ids,         \
        std::vector<uint64_t> &row_ptr,         \
        std::vector<type> &data                 \
    );


#define FORCE_COMPILE(type)                     \
    FORCE_COMPILE_GENERATE_RANDOM_VALUE(type)   \
    FORCE_COMPILE_CSR_SPMV_CPU_KERNEL(type)     \
    FORCE_COMPILE_VERIFY_SPMV_RESULT(type)      \
    FORCE_COMPILE_GENERATE_RANDOM_CSR(type)

FORCE_COMPILE(float)
FORCE_COMPILE(double)

#undef FORCE_COMPILE
