#include <iomanip>
#include <iostream>
#include <fstream>
#include <cuda_fp16.h>
#include <unistd.h>

#include "gpu_error.hpp"
#include "measurement_series.hpp"
#include "cuda_metrics/measureMetricPW.hpp"
#include "cache_flush.hpp"

template <typename dtype, int num_iters, int test_type, bool aligned>
void measure() {
    
}