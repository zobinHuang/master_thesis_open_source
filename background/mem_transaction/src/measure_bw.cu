#include <iomanip>
#include <iostream>
#include <fstream>
#include <cuda_fp16.h>
#include <unistd.h>

#include "gpu_error.hpp"
#include "measurement_series.hpp"
#include "cuda_metrics/measureMetricPW.hpp"
#include "cache_flush.hpp"

enum {
    TEST_TYPE_L1_LD = 1,
    TEST_TYPE_L1_ST,
    TEST_TYPE_L2_LD,
    TEST_TYPE_L2_ST,
    TEST_TYPE_DRAM_LD,
    TEST_TYPE_DRAM_ST
};

template<typename dtype, int num_elements, int num_iters>
__global__ void issue_store(dtype *addr){
    int idx = blockIdx.x * blockDim.x +  threadIdx.x;

    // register definition
    if constexpr (std::is_same<dtype, uint8_t>::value){
        asm(".reg.u8 t0;\n\t");
    }
    else if constexpr (std::is_same<dtype, half>::value){
        asm(".reg.f16 t0;\n\t");
    }
    else if constexpr (std::is_same<dtype, float>::value){
        asm(".reg.f32 t0;\n\t");
    }
    else if constexpr (std::is_same<dtype, double>::value){
        asm(".reg.f64 t0;\n\t");
    }
    else if constexpr (std::is_same<dtype, float4>::value){ 
        asm(".reg.f32 t<4>;\n\t" ".reg.f32 g<4>;\n\t"); 
    }

    #pragma unroll
    for(int j=0; j<num_iters; j++){
        #pragma unroll
        for(int i=idx; i<num_elements; i+=blockDim.x){
            dtype *thread_addr = addr + i;
            if constexpr (std::is_same<dtype, uint8_t>::value){
                asm volatile (
                    "st.global.cg.u8 [%0], t0;\n\t" 
                    :: "l"(thread_addr)
                );
            } else if constexpr (std::is_same<dtype, half>::value){
                asm volatile (
                    "st.global.cg.f16 [%0], t0;\n\t" 
                    :: "l"(thread_addr)
                );
            } else if constexpr (std::is_same<dtype, float>::value){
                asm volatile (
                    "st.global.cg.f32 [%0], t0;\n\t" 
                    :: "l"(thread_addr)
                );
            } else if constexpr (std::is_same<dtype, double>::value){
                asm volatile (
                    "st.global.cg.f64 [%0], t0;\n\t" 
                    :: "l"(thread_addr)
                );
            } else if constexpr (std::is_same<dtype, float4>::value){
                asm volatile (
                    "st.global.cg.f64 [%0], {t0,t1,t2,t3};\n\t" 
                    :: "l"(thread_addr)
                );
            }
        } // for(int i=idx; i<num_elements; i+=blockDim.x)
    } // for(int j=0; j<num_iters; j++)
}

template<typename dtype, int num_elements, int num_iters>
__global__ void issue_load(dtype *addr){
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
    for(int j=0; j<num_iters; j++){
        #pragma unroll
        for(int i=idx; i<num_elements; i+=blockDim.x){
            dtype *thread_addr = addr + i;
            if constexpr (std::is_same<dtype, uint8_t>::value){
                asm volatile (
                    "ld.global.ca.u8 t0, [%0];\n\t" 
                    // this add prevents scoreboard from ignoring the useless ld instruction above
                    "add.u8 t1, t1, t0;\n\t"
                    :: "l"(thread_addr)
                );
            } else if constexpr (std::is_same<dtype, half>::value){
                asm volatile (
                    "ld.global.ca.f16 t0, [%0];\n\t"
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
            } else if constexpr (std::is_same<dtype, double>::value){
                asm(
                    "ld.global.ca.f64 t0, [%0];\n\t"
                    // this add prevents scoreboard from ignoring the useless ld instruction above
                    "add.f64 t1, t1, t0;\n\t"
                    :: "l"(thread_addr)
                );
            } else if constexpr (std::is_same<dtype, float4>::value){
                asm(
                    "ld.global.ca.v4.f32 {t0,t1,t2,t3}, [%0];\n\t"
                    // these adds prevent scoreboard from ignoring the useless ld instruction above
                    "add.f32 g0, g0, t0;\n\t"
                    "add.f32 g1, g1, t1;\n\t"
                    "add.f32 g2, g2, t2;\n\t"
                    "add.f32 g3, g3, t3;\n\t"
                    :: "l"(thread_addr)
                );
            }
        } // for(int i=idx; i<num_elements; i+=blockDim.x)
    } // for(int j=0; j<num_iters; j++)
}

template <typename dtype, int num_elements, int num_iters, bool aligned>
void start_store_test(std::fstream &f, int BLOCK_COUNT, int BLOCK_SIZE){
    MeasurementSeries duration_ms;
    MeasurementSeries gst_transactions, global_store_requests, gst_throughput, gst_transactions_per_request;
    MeasurementSeries l2_global_store_bytes, l2_write_transactions, l2_write_throughput;
    MeasurementSeries dram_write_transactions, dram_write_bytes, dram_write_throughput;
    dtype *addr;
    l2flush l2;
    cudaEvent_t start, stop;
    float single_duration_ms;

    // test eight times to take average
    for (int i = 0; i < 8; i++) {
        // initialize the destination vector
        if constexpr (aligned){
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
        } else {
            GPU_ERROR(cudaMalloc(&addr, sizeof(dtype)*(num_elements+1)));
            std::vector<dtype> array;
            for(int i=0; i<num_elements+1; i++){
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
            GPU_ERROR(cudaMemcpy(addr, array.data(), sizeof(dtype)*(num_elements+1), cudaMemcpyHostToDevice));
        }

        GPU_ERROR(cudaEventCreate(&start));
        GPU_ERROR(cudaEventCreate(&stop));

        // flush l1/l2 cache
        l2.flush();
        GPU_ERROR(cudaDeviceSynchronize());

        // measure duration
        GPU_ERROR(cudaEventRecord(start));
        if constexpr (aligned)
            issue_store<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr);
        else
            issue_store<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr+1);
        GPU_ERROR(cudaEventRecord(stop));
        GPU_ERROR(cudaDeviceSynchronize());
        GPU_ERROR(cudaEventElapsedTime(&single_duration_ms, start, stop));
        duration_ms.add(single_duration_ms);

        // flush l1/l2 cache
        l2.flush();
        GPU_ERROR(cudaDeviceSynchronize());

        // measure sm <-> l1
        measureL1StoreStart();
        if constexpr (aligned)
            issue_store<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr);
        else
            issue_store<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr+1);
        auto metrics = measureL1StoreStop();
        gst_transactions.add(metrics["gst_transactions"]);
        global_store_requests.add(metrics["global_store_requests"]);
        gst_throughput.add(metrics["gst_throughput"]);
        gst_transactions_per_request.add(metrics["gst_transactions_per_request"]);

        // flush l1/l2 cache
        l2.flush();
        GPU_ERROR(cudaDeviceSynchronize());

        // measure l1 <-> l2
        measureL2StoreStart();
        if constexpr (aligned)
            issue_store<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr);
        else
            issue_store<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr+1);
        metrics = measureL2StoreStop();
        l2_global_store_bytes.add(metrics["l2_global_store_bytes"]);
        l2_write_transactions.add(metrics["l2_write_transactions"]);
        l2_write_throughput.add(metrics["l2_write_throughput"]);

        // flush l1/l2 cache
        l2.flush();
        GPU_ERROR(cudaDeviceSynchronize());

        // measure l2 <-> Dram
        measureDramReadStart();
        if constexpr (aligned)
            issue_store<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr);
        else
            issue_store<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr+1);
        metrics = measureDramReadStop();
        dram_write_transactions.add(metrics["dram_write_transactions"]);
        dram_write_bytes.add(metrics["dram_write_bytes"]);
        dram_write_throughput.add(metrics["dram_write_throughput"]);

        GPU_ERROR(cudaFree(addr));
    }

    f   << BLOCK_COUNT << " "
        << BLOCK_SIZE << " "
        << num_elements << " "
        << (aligned ? "true" : "false") << " " 
        << duration_ms.median() << " "
        << gst_transactions.median() << " "
        << global_store_requests.median() << " "
        << gst_throughput.median() << " "
        << gst_transactions_per_request.median() << " "
        << l2_global_store_bytes.median() << " "
        << l2_write_transactions.median() << " "
        << l2_write_throughput.median() << " "
        << dram_write_transactions.median() << " "
        << dram_write_bytes.median() << " "
        << dram_write_throughput.median() << " "
        << std::endl;
}

template <typename dtype, int num_elements, int num_iters, bool aligned>
void start_load_test(std::fstream &f, int BLOCK_COUNT, int BLOCK_SIZE){
    MeasurementSeries duration_ms;
    MeasurementSeries gld_transactions, global_load_requests, gld_throughput, gld_transactions_per_request;
    MeasurementSeries l2_read_transactions, l2_read_throughput, l2_global_load_bytes;
    MeasurementSeries dram_read_transactions, dram_read_bytes, dram_read_throughput;
    dtype *addr;
    l2flush l2;
    cudaEvent_t start, stop;
    float single_duration_ms;

    // test eight times to take average
    for (int i = 0; i < 8; i++) {
        // initialize the destination vector
        if constexpr (aligned){
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
        } else {
            GPU_ERROR(cudaMalloc(&addr, sizeof(dtype)*(num_elements+1)));
            std::vector<dtype> array;
            for(int i=0; i<num_elements+1; i++){
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
            GPU_ERROR(cudaMemcpy(addr, array.data(), sizeof(dtype)*(num_elements+1), cudaMemcpyHostToDevice));
        }

        GPU_ERROR(cudaEventCreate(&start));
        GPU_ERROR(cudaEventCreate(&stop));
        
        // flush l1/l2 cache
        l2.flush();
        GPU_ERROR(cudaDeviceSynchronize());
        
        // measure duration
        GPU_ERROR(cudaEventRecord(start));
        if constexpr (aligned)
            issue_load<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr);
        else
            issue_load<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr+1);
        GPU_ERROR(cudaEventRecord(stop));
        GPU_ERROR(cudaDeviceSynchronize());
        GPU_ERROR(cudaEventElapsedTime(&single_duration_ms, start, stop));
        duration_ms.add(single_duration_ms);

        // flush l1/l2 cache
        l2.flush();
        GPU_ERROR(cudaDeviceSynchronize());

        // measure sm <-> l1
        measureL1LoadStart();
        if constexpr (aligned)
            issue_load<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr);
        else
            issue_load<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr+1);
        auto metrics = measureL1LoadStop();
        gld_transactions.add(metrics["gld_transactions"]);
        global_load_requests.add(metrics["global_load_requests"]);
        gld_throughput.add(metrics["gld_throughput"]);
        gld_transactions_per_request.add(metrics["gld_transactions_per_request"]);
        
        // flush l1/l2 cache
        l2.flush();
        GPU_ERROR(cudaDeviceSynchronize());

        // measure l1 <-> l2
        measureL2LoadStart();
        if constexpr (aligned)
            issue_load<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr);
        else
            issue_load<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr+1);
        metrics = measureL2LoadStop();
        l2_global_load_bytes.add(metrics["l2_global_load_bytes"]);
        l2_read_transactions.add(metrics["l2_read_transactions"]);
        l2_read_throughput.add(metrics["l2_read_throughput"]);

        // flush l1/l2 cache
        l2.flush();
        GPU_ERROR(cudaDeviceSynchronize());

        // measure l2 <-> Dram
        measureDramReadStart();
        if constexpr (aligned)
            issue_load<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr);
        else
            issue_load<dtype, num_elements, num_iters><<<BLOCK_COUNT, BLOCK_SIZE>>>(addr+1);
        metrics = measureDramReadStop();
        dram_read_transactions.add(metrics["dram_read_transactions"]);
        dram_read_bytes.add(metrics["dram_read_bytes"]);
        dram_read_throughput.add(metrics["dram_read_throughput"]);

        GPU_ERROR(cudaEventDestroy(start));
        GPU_ERROR(cudaEventDestroy(stop));
        GPU_ERROR(cudaFree(addr));
    }

    f   << BLOCK_COUNT << " "
        << BLOCK_SIZE << " "
        << num_elements << " "
        << (aligned ? "true" : "false") << " "
        << duration_ms.median() << " "
        << gld_transactions.median() << " "
        << global_load_requests.median() << " "
        << gld_throughput.median() << " "
        << gld_transactions_per_request.median() << " "
        << l2_global_load_bytes.median() << " "
        << l2_read_transactions.median() << " "
        << l2_read_throughput.median() << " "
        << dram_read_transactions.median() << " "
        << dram_read_bytes.median() << " "
        << dram_read_throughput.median() << " "
        << std::endl;
}

template <typename dtype, int num_iters, int test_type, bool aligned>
void measure() {
    int deviceId, smCount;
    cudaDeviceProp prop;
    std::string deviceName;
    std::fstream f;
    char log_file_name[1024] = {0};

    GPU_ERROR(cudaGetDevice(&deviceId));
    GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    deviceName = prop.name;
    smCount = prop.multiProcessorCount;
    // printf("%s\n", deviceName.c_str());
    // return;

    #define START_TEST(NUM_ELE, TEST_TYPE)                                                                          \
    if(TEST_TYPE == TEST_TYPE_L1_LD || TEST_TYPE == TEST_TYPE_L2_LD || TEST_TYPE == TEST_TYPE_DRAM_LD) {            \
        start_load_test<dtype, NUM_ELE, num_iters, aligned>(f, smCount, 256);                                       \
    } else if (TEST_TYPE == TEST_TYPE_L2_ST || TEST_TYPE == TEST_TYPE_DRAM_ST || TEST_TYPE == TEST_TYPE_L1_ST) {    \
        start_store_test<dtype, NUM_ELE, num_iters, aligned>(f, smCount, 256);                                      \
    }
    
    /* ==================== l1-data load test ==================== */
    // exp1: obtain the overall trends
    if(test_type == TEST_TYPE_L1_LD){
        if(deviceName == "Tesla V100-PCIE-32GB"){
            sprintf(
                log_file_name, "../log/measure_bw_l1_ld_overall_%s_(%s).txt",
                aligned ? "aligned" : "misaligned",
                deviceName.c_str()
            );
            f.open(log_file_name, std::ios::out);

            // write header
            f   // basic setup
                << "block_count " << "block_size " << "num_elements " << "aligned " << "duration_ms "
                // global metrics
                << "gld_transactions " << "global_load_requests " 
                << "gld_throughput " << "gld_transactions_per_request "
                // l2 metrics
                << "l2_global_load_bytes " << "l2_read_transactions " << "l2_read_throughput "
                // dram metrics
                << "dram_read_transactions " << "dram_read_bytes " << "dram_read_throughput "
                << std::endl;

            // (0~128KB): 4N <= 128KB => N <= 2**15 = 32768
            START_TEST((1<<0), TEST_TYPE_L1_LD)
            START_TEST((1<<1), TEST_TYPE_L1_LD)
            START_TEST((1<<2), TEST_TYPE_L1_LD)
            START_TEST((1<<3), TEST_TYPE_L1_LD)
            START_TEST((1<<4), TEST_TYPE_L1_LD)
            START_TEST((1<<5), TEST_TYPE_L1_LD)
            START_TEST((1<<6), TEST_TYPE_L1_LD)
            START_TEST((1<<7), TEST_TYPE_L1_LD)
            START_TEST((1<<8), TEST_TYPE_L1_LD)
            START_TEST((1<<9), TEST_TYPE_L1_LD)
            START_TEST((1<<10), TEST_TYPE_L1_LD)
            START_TEST((1<<11), TEST_TYPE_L1_LD)
            START_TEST((1<<12), TEST_TYPE_L1_LD)
            START_TEST((1<<13), TEST_TYPE_L1_LD)
            START_TEST((1<<14), TEST_TYPE_L1_LD)
            START_TEST((1<<15), TEST_TYPE_L1_LD)
            f.close();
        } else {
            std::cout << "undefined GPU: " << deviceName << std::endl;
        } // if(deviceName == "Tesla V100-PCIE-32GB")
    } // if(test_type == TEST_TYPE_L1_LD)

    memset(log_file_name, 0, sizeof(log_file_name));

    // exp2: obtain the partial shape
    if(test_type == TEST_TYPE_L1_LD){
        if(deviceName == "Tesla V100-PCIE-32GB"){
            sprintf(
                log_file_name, "../log/measure_bw_l1_ld_partial_%s_(%s).txt",
                aligned ? "aligned" : "misaligned",
                deviceName.c_str()
            );
            f.open(log_file_name, std::ios::out);

            // write header
            f   // basic setup
                << "block_count " << "block_size " << "num_elements " << "aligned " << "duration_ms "
                // global metrics
                << "gld_transactions " << "global_load_requests " 
                << "gld_throughput " << "gld_transactions_per_request "
                // l2 metrics
                << "l2_global_load_bytes " << "l2_read_transactions " << "l2_read_throughput "
                // dram metrics
                << "dram_read_transactions " << "dram_read_bytes " << "dram_read_throughput "
                << std::endl;

            START_TEST(904, TEST_TYPE_L1_LD) START_TEST(912, TEST_TYPE_L1_LD) START_TEST(920, TEST_TYPE_L1_LD)
            START_TEST(928, TEST_TYPE_L1_LD) START_TEST(936, TEST_TYPE_L1_LD) START_TEST(944, TEST_TYPE_L1_LD)
            START_TEST(952, TEST_TYPE_L1_LD) START_TEST(960, TEST_TYPE_L1_LD) START_TEST(968, TEST_TYPE_L1_LD)
            START_TEST(976, TEST_TYPE_L1_LD) START_TEST(984, TEST_TYPE_L1_LD) START_TEST(992, TEST_TYPE_L1_LD)
            START_TEST(1000, TEST_TYPE_L1_LD) START_TEST(1008, TEST_TYPE_L1_LD) START_TEST(1016, TEST_TYPE_L1_LD)
            START_TEST(1024, TEST_TYPE_L1_LD) START_TEST(1032, TEST_TYPE_L1_LD) START_TEST(1040, TEST_TYPE_L1_LD)
            START_TEST(1048, TEST_TYPE_L1_LD) START_TEST(1056, TEST_TYPE_L1_LD) START_TEST(1064, TEST_TYPE_L1_LD)
            START_TEST(1072, TEST_TYPE_L1_LD) START_TEST(1080, TEST_TYPE_L1_LD) START_TEST(1088, TEST_TYPE_L1_LD)
            START_TEST(1096, TEST_TYPE_L1_LD) START_TEST(1104, TEST_TYPE_L1_LD) START_TEST(1112, TEST_TYPE_L1_LD)
            START_TEST(1120, TEST_TYPE_L1_LD) START_TEST(1128, TEST_TYPE_L1_LD) START_TEST(1136, TEST_TYPE_L1_LD)
            START_TEST(1144, TEST_TYPE_L1_LD) START_TEST(1152, TEST_TYPE_L1_LD)
            f.close();
        } else {
            std::cout << "undefined GPU: " << deviceName << std::endl;
        } // if(deviceName == "Tesla V100-PCIE-32GB")
    } // if(test_type == TEST_TYPE_L1_LD)
    /* ================ end of l1-data load test ================= */

    memset(log_file_name, 0, sizeof(log_file_name));

    /* ==================== l2 load test ==================== */
    // exp1: obtain the overall trends
    if(test_type == TEST_TYPE_L2_LD){
        if(deviceName == "Tesla V100-PCIE-32GB"){
            sprintf(
                log_file_name, "../log/measure_bw_l2_ld_overall_%s_(%s).txt",
                aligned ? "aligned" : "misaligned",
                deviceName.c_str()
            );
            f.open(log_file_name, std::ios::out);

            // write header
            f   // basic setup
                << "block_count " << "block_size " << "num_elements " << "aligned " << "duration_ms "
                // global metrics
                << "gld_transactions " << "global_load_requests " 
                << "gld_throughput " << "gld_transactions_per_request "
                // l2 metrics
                << "l2_global_load_bytes " << "l2_read_transactions " << "l2_read_throughput "
                // dram metrics
                << "dram_read_transactions " << "dram_read_bytes " << "dram_read_throughput "
                << std::endl;

            // === L2 Range ===
            // 128KB ~ 6MB: 
            // 128K <= 4N <= 6M
            // 32 K <= N <= 1.5M
            // 2**15 <= N < 2**21
            START_TEST((1<<14), TEST_TYPE_L2_LD)    // still l1
            START_TEST((1<<15), TEST_TYPE_L2_LD)
            START_TEST((1<<16), TEST_TYPE_L2_LD)
            START_TEST((1<<17), TEST_TYPE_L2_LD)
            START_TEST((1<<18), TEST_TYPE_L2_LD)
            START_TEST((1<<19), TEST_TYPE_L2_LD)
            START_TEST((1<<20), TEST_TYPE_L2_LD)
            START_TEST((1<<21), TEST_TYPE_L2_LD)    // exceed to Dram
            START_TEST((1<<22), TEST_TYPE_L2_LD)    // exceed to Dram
            f.close();
        } else {
            std::cout << "undefined GPU: " << deviceName << std::endl;
        } // if(deviceName == "Tesla V100-PCIE-32GB")
    } // if(test_type == TEST_TYPE_L2_LD)
    /* ================ end of l2 load test ================= */

    memset(log_file_name, 0, sizeof(log_file_name));

    /* ==================== dram load test ==================== */
    // exp1: obtain the overall trends
    if(test_type == TEST_TYPE_DRAM_LD){
        if(deviceName == "Tesla V100-PCIE-32GB"){
            sprintf(
                log_file_name, "../log/measure_bw_dram_ld_overall_%s_(%s).txt",
                aligned ? "aligned" : "misaligned",
                deviceName.c_str()
            );
            f.open(log_file_name, std::ios::out);

            // write header
            f   // basic setup
                << "block_count " << "block_size " << "num_elements " << "aligned " << "duration_ms "
                // global metrics
                << "gld_transactions " << "global_load_requests " 
                << "gld_throughput " << "gld_transactions_per_request "
                // l2 metrics
                << "l2_global_load_bytes " << "l2_read_transactions " << "l2_read_throughput "
                // dram metrics
                << "dram_read_transactions " << "dram_read_bytes " << "dram_read_throughput "
                << std::endl;

            // === DRAM Range ===
            // 4N > 6MB: 
            // 1.5M < N
            // 2**21 <= N
            START_TEST((1<<18), TEST_TYPE_DRAM_LD)    // still l2
            START_TEST((1<<19), TEST_TYPE_DRAM_LD)    // still l2
            START_TEST((1<<20), TEST_TYPE_DRAM_LD)    // still l2
            START_TEST((1<<21), TEST_TYPE_DRAM_LD)
            START_TEST((1<<22), TEST_TYPE_DRAM_LD)
            START_TEST((1<<23), TEST_TYPE_DRAM_LD)
            START_TEST((1<<24), TEST_TYPE_DRAM_LD)
            START_TEST((1<<25), TEST_TYPE_DRAM_LD)
            START_TEST((1<<26), TEST_TYPE_DRAM_LD)
            f.close();
        } else {
            std::cout << "undefined GPU: " << deviceName << std::endl;
        } // if(deviceName == "Tesla V100-PCIE-32GB")
    } // if(test_type == TEST_TYPE_L2_LD)
    /* ================ end of dram load test ================= */

    memset(log_file_name, 0, sizeof(log_file_name));
    
    /* ==================== l1 store test =================== */
    if(test_type == TEST_TYPE_L1_ST){
        if(deviceName == "Tesla V100-PCIE-32GB"){
            sprintf(
                log_file_name, "../log/measure_bw_l1_st_%s_(%s).txt",
                aligned ? "aligned" : "misaligned",
                deviceName.c_str()
            );
            f.open(log_file_name, std::ios::out);

            // write header
            f   // basic setup
                << "block_count " << "block_size " << "num_elements " << "aligned " << "duration_ms "
                // global metrics
                << "gst_transactions " << "global_store_requests " 
                << "gst_throughput " << "gst_transactions_per_request "
                // l2 metrics
                << "l2_global_store_bytes " << "l2_write_transactions " << "l2_write_throughput "
                // dram metrics
                << "dram_write_transactions " << "dram_write_bytes " << "dram_write_throughput "
                << std::endl;

            
            // 4B <= 4N <= 128KB
            // 1 <= N <= 32768
            // 2**0 <= N < 2**15
            START_TEST((1<<0), TEST_TYPE_L2_ST)
            START_TEST((1<<1), TEST_TYPE_L2_ST)
            START_TEST((1<<2), TEST_TYPE_L2_ST)
            START_TEST((1<<3), TEST_TYPE_L2_ST)
            START_TEST((1<<4), TEST_TYPE_L2_ST)
            START_TEST((1<<5), TEST_TYPE_L2_ST)
            START_TEST((1<<6), TEST_TYPE_L2_ST)
            START_TEST((1<<7), TEST_TYPE_L2_ST)
            START_TEST((1<<8), TEST_TYPE_L2_ST)
            START_TEST((1<<9), TEST_TYPE_L2_ST)
            START_TEST((1<<10), TEST_TYPE_L2_ST)
            START_TEST((1<<11), TEST_TYPE_L2_ST)
            START_TEST((1<<12), TEST_TYPE_L2_ST)
            START_TEST((1<<13), TEST_TYPE_L2_ST)
            START_TEST((1<<14), TEST_TYPE_L2_ST)
            START_TEST((1<<15), TEST_TYPE_L2_ST)
            START_TEST((1<<16), TEST_TYPE_L2_ST)    // exceed to l2
            START_TEST((1<<17), TEST_TYPE_L2_ST)    // exceed to l2
            f.close();
        } else {
            std::cout << "undefined GPU: " << deviceName << std::endl;
        } // if(deviceName == "Tesla V100-PCIE-32GB")
    } // if(test_type == TEST_TYPE_L1_ST)
    /* ================ end of l1 store test ================ */

    memset(log_file_name, 0, sizeof(log_file_name));

    /* ==================== l2 store test =================== */
    if(test_type == TEST_TYPE_L2_ST){
        if(deviceName == "Tesla V100-PCIE-32GB"){
            sprintf(
                log_file_name, "../log/measure_bw_l2_st_%s_(%s).txt",
                aligned ? "aligned" : "misaligned",
                deviceName.c_str()
            );
            f.open(log_file_name, std::ios::out);

            // write header
            f   // basic setup
                << "block_count " << "block_size " << "num_elements " << "aligned " << "duration_ms "
                // global metrics
                << "gst_transactions " << "global_store_requests " 
                << "gst_throughput " << "gst_transactions_per_request "
                // l2 metrics
                << "l2_global_store_bytes " << "l2_write_transactions " << "l2_write_throughput "
                // dram metrics
                << "dram_write_transactions " << "dram_write_bytes " << "dram_write_throughput "
                << std::endl;

            
            // 128KB <= 4N <= 6MB
            // 32768 <= N <= 1.5M
            // 2**15 <= N < 2**21
            START_TEST((1<<14), TEST_TYPE_L2_ST)    // still l1-data
            START_TEST((1<<15), TEST_TYPE_L2_ST)    // still l1-data
            START_TEST((1<<16), TEST_TYPE_L2_ST)
            START_TEST((1<<17), TEST_TYPE_L2_ST)
            START_TEST((1<<18), TEST_TYPE_L2_ST)
            START_TEST((1<<19), TEST_TYPE_L2_ST)
            START_TEST((1<<20), TEST_TYPE_L2_ST)
            START_TEST((1<<21), TEST_TYPE_L2_ST)    // exceed to Dram
            START_TEST((1<<22), TEST_TYPE_L2_ST)    // exceed to Dram
            f.close();
        } else {
            std::cout << "undefined GPU: " << deviceName << std::endl;
        } // if(deviceName == "Tesla V100-PCIE-32GB")
    } // if(test_type == TEST_TYPE_L2_ST)
    /* ================ end of l2 store test ================ */

    memset(log_file_name, 0, sizeof(log_file_name));

    /* ==================== dram store test =================== */
    if(test_type == TEST_TYPE_DRAM_ST){
        if(deviceName == "Tesla V100-PCIE-32GB"){
            sprintf(
                log_file_name, "../log/measure_bw_dram_st_%s_(%s).txt",
                aligned ? "aligned" : "misaligned",
                deviceName.c_str()
            );
            f.open(log_file_name, std::ios::out);

            // write header
            f   // basic setup
                << "block_count " << "block_size " << "num_elements " << "aligned " << "duration_ms "
                // global metrics
                << "gst_transactions " << "global_store_requests " 
                << "gst_throughput " << "gst_transactions_per_request "
                // l2 metrics
                << "l2_global_store_bytes " << "l2_write_transactions " << "l2_write_throughput "
                // dram metrics
                << "dram_write_transactions " << "dram_write_bytes " << "dram_write_throughput "
                << std::endl;

            
            // === DRAM Range ===
            // 4N > 6MB: 
            // 1.5M < N
            // 2**21 <= N
            START_TEST((1<<18), TEST_TYPE_DRAM_ST)    // still l2
            START_TEST((1<<19), TEST_TYPE_DRAM_ST)    // still l2
            START_TEST((1<<20), TEST_TYPE_DRAM_ST)
            START_TEST((1<<21), TEST_TYPE_DRAM_ST)
            START_TEST((1<<22), TEST_TYPE_DRAM_ST)
            START_TEST((1<<23), TEST_TYPE_DRAM_ST)
            START_TEST((1<<24), TEST_TYPE_DRAM_ST)
            START_TEST((1<<25), TEST_TYPE_DRAM_ST)
            START_TEST((1<<26), TEST_TYPE_DRAM_ST)
            f.close();
        } else {
            std::cout << "undefined GPU: " << deviceName << std::endl;
        } // if(deviceName == "Tesla V100-PCIE-32GB")
    } // if(test_type == TEST_TYPE_L2_ST)
    /* ================ end of dram store test ================ */

    #undef START_TEST
}

int main(){
    GPU_ERROR(cudaSetDevice(0));

    // aligned test
    measure</* dtype */ float, /* num_iters */ (1<<3), /* test_type */ TEST_TYPE_L1_LD, /* aligned */ true>();
    measure</* dtype */ float, /* num_iters */ (1<<3), /* test_type */ TEST_TYPE_L2_LD, /* aligned */ true>();
    measure</* dtype */ float, /* num_iters */ (1<<3), /* test_type */ TEST_TYPE_DRAM_LD, /* aligned */ true>();
    measure</* dtype */ float, /* num_iters */ 1, /* test_type */ TEST_TYPE_L1_ST, /* aligned */ true>();
    measure</* dtype */ float, /* num_iters */ 1, /* test_type */ TEST_TYPE_L2_ST, /* aligned */ true>();
    measure</* dtype */ float, /* num_iters */ 1, /* test_type */ TEST_TYPE_DRAM_ST, /* aligned */ true>();

    // misaligned test
    measure</* dtype */ float, /* num_iters */ (1<<3), /* test_type */ TEST_TYPE_L1_LD, /* aligned */ false>();
    measure</* dtype */ float, /* num_iters */ (1<<3), /* test_type */ TEST_TYPE_L2_LD, /* aligned */ false>();
    measure</* dtype */ float, /* num_iters */ (1<<3), /* test_type */ TEST_TYPE_DRAM_LD, /* aligned */ false>();
    measure</* dtype */ float, /* num_iters */ 1, /* test_type */ TEST_TYPE_L1_ST, /* aligned */ false>();
    measure</* dtype */ float, /* num_iters */ 1, /* test_type */ TEST_TYPE_L2_ST, /* aligned */ false>();
    measure</* dtype */ float, /* num_iters */ 1, /* test_type */ TEST_TYPE_DRAM_ST, /* aligned */ false>();

    return 0;
}
