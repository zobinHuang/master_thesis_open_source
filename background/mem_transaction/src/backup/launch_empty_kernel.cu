#include <iomanip>
#include <iostream>
#include <fstream>
#include <cuda_fp16.h>

#include "gpu_error.hpp"
#include "measurement_series.hpp"
#include "cuda_metrics/measureMetricPW.hpp"
#include "cache_flush.hpp"
#include "time.hpp"


template<typename dtype>       
__global__ void kernel_1(dtype *ds_1){}

template<typename dtype>       
__global__ void kernel_2(dtype *ds_1, dtype *ds_2){}

template<typename dtype>       
__global__ void kernel_3(dtype *ds_1, dtype *ds_2, dtype *ds_3){}

template<typename dtype>       
__global__ void kernel_4(dtype *ds_1, dtype *ds_2, dtype *ds_3, dtype *ds_4){}

template<typename dtype>       
__global__ void kernel_5(dtype *ds_1, dtype *ds_2, dtype *ds_3, dtype *ds_4, dtype *ds_5){}

template<typename dtype>       
__global__ void kernel_6(dtype *ds_1, dtype *ds_2, dtype *ds_3, dtype *ds_4, dtype *ds_5, dtype *ds_6){}

template<typename dtype>       
__global__ void kernel_7(dtype *ds_1, dtype *ds_2, dtype *ds_3, dtype *ds_4, dtype *ds_5, dtype *ds_6, dtype *ds_7){}

template<typename dtype>       
__global__ void kernel_8(dtype *ds_1, dtype *ds_2, dtype *ds_3, dtype *ds_4, dtype *ds_5, dtype *ds_6, dtype *ds_7, dtype *ds_8){}

template<typename dtype>
void launch_kernel(dtype *addr, int blockCount, int blockSize, int num_params){
    switch (num_params)
    {
    case 1:
        kernel_1<dtype><<<blockCount, blockSize>>>(addr);
        break;

    case 2:
        kernel_2<dtype><<<blockCount, blockSize>>>(addr, addr);
        break;

    case 3:
        kernel_3<dtype><<<blockCount, blockSize>>>(addr, addr, addr);
        break;

    case 4:
        kernel_4<dtype><<<blockCount, blockSize>>>(addr, addr, addr, addr);
        break;

    case 5:
        kernel_5<dtype><<<blockCount, blockSize>>>(addr, addr, addr, addr, addr);
        break;

    case 6:
        kernel_6<dtype><<<blockCount, blockSize>>>(addr, addr, addr, addr, addr, addr);
        break;

    case 7:
        kernel_7<dtype><<<blockCount, blockSize>>>(addr, addr, addr, addr, addr, addr, addr);
        break;

    case 8:
        kernel_8<dtype><<<blockCount, blockSize>>>(addr, addr, addr, addr, addr, addr, addr, addr);
        break;
    
    default:
        break;
    }
}

template <typename dtype>
void measure() {
    int i, deviceId, smCount, blockCount, numParams;
    int blockSize = 32;
    cudaDeviceProp prop;
    std::string deviceName;
    dtype *ds;
    uint64_t num_load_bytes;
    int maxActiveBlocks = 0;
    l2flush l2;
    std::fstream f;
	cudaEvent_t start, stop;
    uint64_t start_ns, stop_ns;
    float time_ms;
    
    f.open("../log/launch_empty_kernel.txt", std::ios::out);

    GPU_ERROR(cudaGetDevice(&deviceId));
    GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    deviceName = prop.name;
    smCount = prop.multiProcessorCount;

    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, kernel_1<dtype>, blockSize, 0));

    for(blockCount=1; blockCount < smCount*maxActiveBlocks; blockCount*=2){
        for(blockSize=32; blockSize<=256; blockSize*=2){
            for(numParams=1; numParams<=8; numParams+=1){
                MeasurementSeries dram_load_bytes;
                MeasurementSeries dram_read_transactions;
                MeasurementSeries duration;

                for(int i=0; i< 16; i++){
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);

                    l2.flush();
                    GPU_ERROR(cudaDeviceSynchronize());

                    measureDramReadStart();
                    launch_kernel<dtype>(ds, blockCount, blockSize, numParams);
                    auto metrics = measureDramReadStop();
                    dram_load_bytes.add(metrics[0]);
                    dram_read_transactions.add(metrics[1]);
                    l2.flush();
                    GPU_ERROR(cudaDeviceSynchronize());

                    GPU_ERROR(cudaEventRecord(start));
                    start_ns = get_time_ns();
                    launch_kernel<dtype>(ds, blockCount, blockSize, numParams);
                    cudaDeviceSynchronize();
                    stop_ns = get_time_ns();
                    GPU_ERROR(cudaEventRecord(stop));
                    
                    duration.add((double)(stop_ns-start_ns)/1000.0f);

                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);
                }

                f << blockCount << " "
                    << blockSize << " "
                    << numParams << " "
                    << duration.median() << " "
                    << dram_load_bytes.median() << " "
                    << dram_read_transactions.median()
                    << std::endl;
            }
        }
    }
}

int main(){
    GPU_ERROR(cudaSetDevice(1));
    measure</* dtype */ float>();
    return 0;
}