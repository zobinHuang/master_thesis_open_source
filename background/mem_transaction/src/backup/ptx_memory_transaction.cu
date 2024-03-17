#include <iomanip>
#include <iostream>
#include <cuda.h>
#include <stdint.h>
#include <cuda_fp16.h>

/* half kernel */
template<typename dtype, int num_elements, typename std::enable_if<std::is_same<dtype, uint16_t>::value,int>::type = 0>
__global__ void kernel(dtype *a1){
    dtype d1;
    #pragma unroll
    for(int i=0; i<num_elements; i++){
        asm("ld.global.cg.u16 %0, [%1];" : "=h"(d1) : "l"(a1)); /* bypass l1 cache */
        if(d1){
            asm("st.global.cg.u16 [%0], %1;" : "=l"(a1) : "h"(d1)); /* bypass l1 cache */
        }
    }
}

/* float kernel */
template<typename dtype, int num_elements, typename std::enable_if<std::is_same<dtype, float>::value,int>::type = 0>
__global__ void kernel(dtype *a1){
    dtype d1;
    #pragma unroll
    for(int i=0; i<num_elements; i++){
        asm("ld.global.cg.f32 %0, [%1];" : "=f"(d1) : "l"(a1)); /* bypass l1 cache */
        if(d1){
            asm("st.global.cg.f32 [%0], %1;" : "=l"(a1) : "f"(d1)); /* bypass l1 cache */
        }
    }
}

/* double kernel */
template<typename dtype, int num_elements, typename std::enable_if<std::is_same<dtype, double>::value,int>::type = 0>
__global__ void kernel(dtype *a1){
    dtype d1;
    #pragma unroll
    for(int i=0; i<num_elements; i++){
        asm("ld.global.cg.f64 %0, [%1];" : "=d"(d1) : "l"(a1)); /* bypass l1 cache */
        if(d1){
            asm("st.global.cg.f64 [%0], %1;" : "=l"(a1) : "d"(d1)); /* bypass l1 cache */
        }
    }
}

int main(){
    kernel</* dtype */ float, /* num_elements */ 1><<<1,32>>>(NULL);
    return 0;
}