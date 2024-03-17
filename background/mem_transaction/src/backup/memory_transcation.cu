#include <iomanip>
#include <iostream>
#include <fstream>
#include <cuda_fp16.h>

#include "gpu_error.hpp"
#include "measurement_series.hpp"
#include "cuda_metrics/measureMetricPW.hpp"
#include "cache_flush.hpp"

// nvcc --ptx ../src/ptx_memory_transaction.cu -o ../src/ptx_memory_transaction.o

/* uint16_t kernel */
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

template <typename dtype, int num_elements>
void launch_kernel(dtype *addr, int blockCount, int blockSize){
    kernel<dtype, num_elements><<<blockCount, blockSize>>>(addr);
}

template <typename dtype, int num_elements>
void measure() {
    int i, deviceId, smCount, blockCount;
    int blockSize = 32;
    cudaDeviceProp prop;
    std::string deviceName;
    dtype *ds;
    uint64_t num_load_bytes;
    int maxActiveBlocks = 0;
    l2flush l2;
    std::fstream f;

    f.open("../log/memory_trasaction.txt", std::ios::out);

    GPU_ERROR(cudaGetDevice(&deviceId));
    GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    deviceName = prop.name;
    smCount = prop.multiProcessorCount;

    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, kernel<dtype, num_elements>, blockSize, 0)
    );

    GPU_ERROR(cudaMalloc(&ds, num_elements * sizeof(dtype)));
    GPU_ERROR(cudaDeviceSynchronize());
    l2.flush();
    GPU_ERROR(cudaDeviceSynchronize());

    // deploy #maxActiveBlocks blocks per sm
    for(blockCount=1; blockCount < smCount*maxActiveBlocks; blockCount*=2){
        for(blockSize=32; blockSize<=256; blockSize*=2){
            MeasurementSeries load_transactions;
            MeasurementSeries load_requests;
            MeasurementSeries load_transactions_per_request;
            MeasurementSeries dram_read_bytes;
            MeasurementSeries dram_read_transactions;
            MeasurementSeries l2_load_bytes;

            for(i=0; i<16; i++){
                l2.flush();
                GPU_ERROR(cudaDeviceSynchronize());

                measureDRAMBytesStart();
                launch_kernel<dtype, num_elements>(ds, blockCount, blockSize);
                auto metrics = measureDRAMBytesStop();
                dram_read_bytes.add(metrics[0]);
                l2.flush();
                GPU_ERROR(cudaDeviceSynchronize());

                measureDramReadTransactionStart();
                launch_kernel<dtype, num_elements>(ds, blockCount, blockSize);
                metrics = measureDramReadTransactionStop();
                dram_read_transactions.add(metrics[0]);
                l2.flush();
                GPU_ERROR(cudaDeviceSynchronize());

                measureL2BytesStart();
                launch_kernel<dtype, num_elements>(ds, blockCount, blockSize);
                metrics = measureL2BytesStop();
                l2_load_bytes.add(metrics[0]);
                l2.flush();
                GPU_ERROR(cudaDeviceSynchronize());

                measureGlobalLoadTranscationsStart();
                launch_kernel<dtype, num_elements>(ds, blockCount, blockSize);
                metrics = measureGlobalLoadTranscationsStop();
                load_requests.add(metrics[0]);
                load_transactions.add(metrics[1]);
                load_transactions_per_request.add(metrics[2]);                
            }

            f << blockCount << " "
                << blockSize << " "
                << dram_read_bytes.median() << " "
                << dram_read_transactions.median() << " "
                << l2_load_bytes.median() << " "
                << load_requests.median() << " "
                << load_transactions.median() << " "
                << load_transactions_per_request.median() << " "
                << std::endl;
            }
    }
}

int main(){
    GPU_ERROR(cudaSetDevice(1));
    measure</* dtype */ float, /* num_elements */ 1>();
    return 0;
}
