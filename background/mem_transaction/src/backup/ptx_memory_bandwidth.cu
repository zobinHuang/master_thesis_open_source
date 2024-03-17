#include <iomanip>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_fp16.h>

template<typename dtype, int num_elements>
__global__ void measure_l1bw(dtype *addr){
    int idx = blockIdx.x * blockDim.x +  threadIdx.x;

    // register definition
    if constexpr (std::is_same<dtype, uint8_t>::value){ asm(".reg.u8 testreg;"); }
    else if constexpr (std::is_same<dtype, half>::value){ asm(".reg.f16 testreg;"); }
    else if constexpr (std::is_same<dtype, float>::value){ asm(".reg.f32 testreg;"); }
    else if constexpr (std::is_same<dtype, double>::value){ asm(".reg.f64 testreg;"); }
    else if constexpr (std::is_same<dtype, float4>::value){ asm(".reg.f32 testreg<4>;"); }

    // load initial value from L1 (expected)
    #pragma unroll
    for(int i=idx; i<num_elements; i+=blockDim.x){
        dtype *thread_addr = addr + i;
        if constexpr (std::is_same<dtype, uint8_t>::value){
            asm("ld.global.ca.u8 testreg, [%0];" :: "l"(thread_addr));
        } else if constexpr (std::is_same<dtype, half>::value){
            asm("ld.global.ca.f16 testreg, [%0];" :: "l"(thread_addr));
        } else if constexpr (std::is_same<dtype, float>::value){
            // asm ("ld.global.f32 testreg, [%0];" :: "l"(thread_addr));
            asm volatile ("ld.global.f32 r1, [%0];" :: "l"(thread_addr) : "memory");
        } else if constexpr (std::is_same<dtype, double>::value){
            asm("ld.global.ca.f64 testreg, [%0];" :: "l"(thread_addr));
        } else if constexpr (std::is_same<dtype, float4>::value){
            asm("ld.global.ca.v4.f32 {testreg0,testreg1,testreg2,testreg3}, [%0];" :: "l"(thread_addr));
        }
    }
}

int main(){
    measure_l1bw<float, 1><<<1,32>>>(NULL);
}