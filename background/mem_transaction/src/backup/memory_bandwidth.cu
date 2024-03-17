#include <iomanip>
#include <iostream>
#include <fstream>
#include <cuda_fp16.h>
#include <unistd.h>

#include "gpu_error.hpp"
#include "measurement_series.hpp"
#include "cuda_metrics/measureMetricPW.hpp"
#include "cache_flush.hpp"

template<typename dtype, int num_elements>
__global__ void init_measure_l1bw(dtype *addr){
    int idx = blockIdx.x * blockDim.x +  threadIdx.x;

    // register definition
    if constexpr (std::is_same<dtype, uint8_t>::value){
        asm(".reg.u8 t<2>;\n\t");
    }
    else if constexpr (std::is_same<dtype, half>::value){
        asm(".reg.f16 t<2>;\n\t");
    }
    else if constexpr (std::is_same<dtype, float>::value){
        asm(".reg.f32 t<2>;\n\t");
    }
    else if constexpr (std::is_same<dtype, double>::value){
        asm(".reg.f64 t<2>;\n\t");
    }
    else if constexpr (std::is_same<dtype, float4>::value){ 
        asm(".reg.f32 t<4>;\n\t" ".reg.f32 g<4>;\n\t"); 
    }

    // store initial value (bypassing L1)
    // #pragma unroll
    // for(int i=idx; i<num_elements; i+=blockDim.x){
    //     dtype *thread_addr = addr + i;
    //     if constexpr (std::is_same<dtype, uint8_t>::value){
    //         asm("st.global.cg.u8 [%0], testreg;\n\t" : "=l"(thread_addr));
    //     } else if constexpr (std::is_same<dtype, half>::value){
    //         asm("st.global.cg.f16 [%0], testreg;\n\t" : "=l"(thread_addr));
    //     } else if constexpr (std::is_same<dtype, float>::value){
    //         asm("st.global.cg.f32 [%0], testreg;\n\t" : "=l"(thread_addr));
    //     } else if constexpr (std::is_same<dtype, double>::value){
    //         asm("st.global.cg.f64 [%0], testreg;\n\t" : "=l"(thread_addr));
    //     } else if constexpr (std::is_same<dtype, float4>::value){
    //         asm("st.global.cg.v4.f32 [%0], {testreg0,testreg1,testreg2,testreg3};\n\t" : "=l"(thread_addr));
    //     }
    // }

    // // load initial value to activate L1
    // #pragma unroll
    // for(int i=idx; i<num_elements; i+=blockDim.x){
    //     dtype *thread_addr = addr + i;
    //     if constexpr (std::is_same<dtype, uint8_t>::value){
    //         asm("ld.global.ca.u8 testreg, [%0];\n\t" :: "l"(thread_addr));
    //     } else if constexpr (std::is_same<dtype, half>::value){
    //         asm("ld.global.ca.f16 testreg, [%0];\n\t" :: "l"(thread_addr));
    //     } else if constexpr (std::is_same<dtype, float>::value){
    //         asm("ld.global.ca.f32 testreg, [%0];\n\t" :: "l"(thread_addr));
    //     } else if constexpr (std::is_same<dtype, double>::value){
    //         asm("ld.global.ca.f64 testreg, [%0];\n\t" :: "l"(thread_addr));
    //     } else if constexpr (std::is_same<dtype, float4>::value){
    //         asm("ld.global.ca.v4.f32 {testreg0,testreg1,testreg2,testreg3}, [%0];\n\t" :: "l"(thread_addr));
    //     }
    // }
}

template<typename dtype, int num_elements>
__global__ void measure_l1bw(dtype *addr){
    int idx = blockIdx.x * blockDim.x +  threadIdx.x;

    // register definition
    if constexpr (std::is_same<dtype, uint8_t>::value){
        asm(".reg.u8 t<2>;\n\t");
    }
    else if constexpr (std::is_same<dtype, half>::value){
        asm(".reg.f16 t<2>;\n\t");
    }
    else if constexpr (std::is_same<dtype, float>::value){
        asm(".reg.f32 t<2>;\n\t");
    }
    else if constexpr (std::is_same<dtype, double>::value){
        asm(".reg.f64 t<2>;\n\t");
    }
    else if constexpr (std::is_same<dtype, float4>::value){ 
        asm(".reg.f32 t<4>;\n\t" ".reg.f32 g<4>;\n\t"); 
    }

    #pragma unroll
    for(int i=idx; i<num_elements; i+=blockDim.x){
        dtype d1;
        dtype *thread_addr = addr + i;

        if constexpr (std::is_same<dtype, uint8_t>::value){
            asm volatile (
                "ld.global.ca.u8 t0, [%0];" 
                // this add prevents scoreboard from ignoring the useless ld instruction above
                "add.u8 t1, t1, t0;\n\t"
                :: "l"(thread_addr)
            );
        } else if constexpr (std::is_same<dtype, half>::value){
            asm volatile (
                "ld.global.ca.f16 t0, [%0];"
                // this add prevents scoreboard from ignoring the useless ld instruction above
                "add.f16 t1, t1, t0;\n\t"
                :: "l"(thread_addr)
            );
        } else if constexpr (std::is_same<dtype, float>::value){
            asm volatile (
                "ld.global.ca.f32 t0, [%0];\n\t"
                // this add prevents scoreboard from ignoring the useless ld instruction above
                "add.f32 t1, t1, t0;\n\t"
                :: "l"(thread_addr) : "memory"
            );
            // asm("ld.global.ca.f32 %0, [%1];" : "=f"(d1) :  "l"(thread_addr));
        } else if constexpr (std::is_same<dtype, double>::value){
            asm(
                "ld.global.ca.f64 t0, [%0];\n\t"
                // this add prevents scoreboard from ignoring the useless ld instruction above
                "add.f64 t1, t1, t0;\n\t"
                :: "l"(thread_addr)
            );
        } else if constexpr (std::is_same<dtype, float4>::value){
            asm(
                "ld.global.ca.v4.f32 {t0,t1,t2,t3}, [%0];"
                // these adds prevent scoreboard from ignoring the useless ld instruction above
                "add.f32 g0, g0, t0;\n\t"
                "add.f32 g1, g1, t1;\n\t"
                "add.f32 g2, g2, t2;\n\t"
                "add.f32 g3, g3, t3;\n\t"
                :: "l"(thread_addr)
            );
        }
    }
}

template <typename dtype, int num_elements>
void test_l1bw( int blockCount, int blockSize, std::fstream &f){
    MeasurementSeries load_transactions;
    MeasurementSeries load_requests;
    MeasurementSeries load_transactions_per_request;
    MeasurementSeries dram_read_bytes;
    MeasurementSeries dram_read_transactions;
    MeasurementSeries l2_load_transactions;
    MeasurementSeries l1_hit_rate;
    MeasurementSeries duration_ms;
    cudaEvent_t start, stop;
    float time_ms;
    dtype *addr;
    l2flush l2;

    for (int i = 0; i < 8; i++) {
        GPU_ERROR(cudaEventCreate(&start));
        GPU_ERROR(cudaEventCreate(&stop));
        GPU_ERROR(cudaMalloc(&addr, sizeof(dtype)*num_elements));
        std::vector<dtype> array;
        for(int i=0; i<num_elements; i++){
            if constexpr (std::is_same<dtype, uint8_t>::value){
                array.push_back(1);
            } else if constexpr (std::is_same<dtype, float4>::value){
                float4 *f = new float4();
                f->x = 1.0f; f->y = 1.0f; f->z = 1.0f; f->w = 1.0f;
                array.push_back(*f);
            } else {
                array.push_back(static_cast<dtype>(1.0f));
            }
        }
            
        GPU_ERROR(cudaMemcpy(addr, array.data(), sizeof(dtype)*num_elements, cudaMemcpyHostToDevice));

        // init_measure_l1bw<dtype, num_elements><<<blockCount, blockSize>>>(addr);
        // GPU_ERROR(cudaDeviceSynchronize());

        // measure load transaction/request statistics
        measureGlobalLoadStart();
        measure_l1bw<dtype, num_elements><<<blockCount, blockSize>>>(addr);
        auto metrics = measureGlobalLoadStop();
        load_requests.add(metrics[0]);
        load_transactions.add(metrics[1]);
        load_transactions_per_request.add(metrics[2]);

        // measure duration
        GPU_ERROR(cudaEventRecord(start));
        measure_l1bw<dtype, num_elements><<<blockCount, blockSize>>>(addr);
        GPU_ERROR(cudaEventRecord(stop));
        GPU_ERROR(cudaDeviceSynchronize());
        GPU_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
        duration_ms.add(time_ms);

        // make sure dram is basically free from reading
        measureDramReadStart();
        measure_l1bw<dtype, num_elements><<<blockCount, blockSize>>>(addr);
        metrics = measureDramReadStop();
        dram_read_bytes.add(metrics[0]);
        dram_read_transactions.add(metrics[1]);

        // make sure l2 cache is basically free from loading
        measureL2LoadStart();
        measure_l1bw<dtype, num_elements><<<blockCount, blockSize>>>(addr);
        metrics = measureL2LoadStop();
        l2_load_transactions.add(metrics[0]);

        GPU_ERROR(cudaFree(addr));
        GPU_ERROR(cudaEventDestroy(start));
        GPU_ERROR(cudaEventDestroy(stop));
    }

    f   << blockCount << " "
        << blockSize << " "
        << load_transactions.median() << " "
        << load_requests.median() << " "
        << load_transactions_per_request.median() << " "
        << dram_read_bytes.median() << " "
        << dram_read_transactions.median() << " "
        << l2_load_transactions.median() << " "
        << duration_ms.median() << " "
        << l1_hit_rate.median() << " "
        << std::endl;
}

template <typename dtype, int num_elements>
void measure() {
    int deviceId, smCount, blockCount, blockSize;
    cudaDeviceProp prop;
    std::string deviceName;
    int maxActiveBlocks = 0;
    std::fstream f;

    f.open("../log/memory_bandwidth.txt", std::ios::out);

    GPU_ERROR(cudaGetDevice(&deviceId));
    GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    deviceName = prop.name;
    smCount = prop.multiProcessorCount;

    for(blockSize=32; blockSize<=256; blockSize*=2){
        // get maximum block per sm
        GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, measure_l1bw<dtype, num_elements>, blockSize, 0)
        );
        
        for(blockCount=1; blockCount < smCount*maxActiveBlocks; blockCount*=2){
            test_l1bw<dtype, num_elements>(blockCount, blockSize, f);
        }
    }
}

int main(){
    GPU_ERROR(cudaSetDevice(0));
    measure</* dtype */ float, /* num_elements */ 64>();
    return 0;
}
