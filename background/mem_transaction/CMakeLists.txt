# cmake version
cmake_minimum_required(VERSION 3.16.3)

# project info
project(nvlink_perf LANGUAGES CXX CUDA)

# set executable output path
set(PATH_EXECUTABLE bin)
execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ../${PATH_EXECUTABLE})
SET(EXECUTABLE_OUTPUT_PATH ../${PATH_EXECUTABLE})

# set library output path
set(PATH_LIBRARY lib)
execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ../${PATH_LIBRARY})
SET(LIBRARY_OUTPUT_PATH ../${PATH_LIBRARY})

# use cmake module to detect CUDA
find_package(CUDA)
find_package(CUDAToolkit)

# use cmake module to detect Boost
include (third_party/preprocessor/CMakeLists.txt)

# ====================== PROFILING PROGRAM ======================
set(SOURCE_DIR src)
add_executable(memory_testsuite ${SOURCE_DIR}/memory_testsuite.cu)
target_compile_options(
    memory_testsuite PRIVATE
    # disable all nvcc optimizations
    $<$<COMPILE_LANGUAGE:CUDA>:-g -G -Xcompiler -O0 -Xptxas -O0 -lineinfo -O0>
)

add_executable(measure_bw ${SOURCE_DIR}/measure_bw.cu)
target_compile_options(
    measure_bw PRIVATE
    # disable all nvcc optimizations
    $<$<COMPILE_LANGUAGE:CUDA>:-g -G -Xcompiler -O0 -Xptxas -O0 -lineinfo -O0>
)

# >>> global configuration
set(PROFILING_TARGETS 
    memory_testsuite
    measure_bw
)
foreach( profiling_target ${PROFILING_TARGETS} )
    # link library
    target_link_libraries(${profiling_target} PUBLIC
        CUDA::nvToolsExt
        boost_preprocessor
        -lcupti -lcuda -lnvidia-ml -lnvperf_host -lnvperf_target -lstdc++fs
    )

    # add dynamic linking path
    target_link_directories(
        ${profiling_target} PUBLIC 
        ${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/lib64  # CUPTI
    )
    
    # add include path
    target_include_directories(
        ${profiling_target} PUBLIC include                  # local include
        ${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/include    # CUPTI
        third_party/preprocessor/include                    # boost::preprocessor
    )

    # set CUDA architecture for target
    # a100: sm_80, compute_80
    # 3090: sm_86, compute_86
    # v100: sm_70, compute_70
    # set_target_properties(${profiling_target} PROPERTIES CUDA_ARCHITECTURES "70;80;86")
    set_target_properties(${profiling_target} PROPERTIES CUDA_ARCHITECTURES "70")

    # add compiling options
    target_compile_features(${profiling_target} PUBLIC cxx_std_17)
    # target_compile_options(
    #     ${profiling_target} PRIVATE 
    #     # a100: sm_80, compute_80
    #     # 3090: sm_86, compute_86
    #     # v100: sm_70, compute_70
    #     $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas="-v" -gencode arch=compute_80,code=sm_80>
    # )
endforeach( profiling_target ${PROFILING_TARGETS} )
