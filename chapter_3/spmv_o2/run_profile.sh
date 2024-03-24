declare -a m_array=(64 256 512 1024 2048 4096 8192 16384 32768 65536)
declare -a n_array=(64 256 512 1024)
declare -a r_array=(0.5 0.2 0.1 0.01)
declare -a d_array=(1 4 16 32)

for m in "${m_array[@]}"
do
    for n in "${n_array[@]}"
    do
        for r in "${r_array[@]}"
        do
            for d in "${d_array[@]}"
            do
                sudo /usr/local/cuda-12.1/bin/ncu --target-processes all ../bin/spmv_csr                                            \
                    -m $m -n $n -r $r -d $d                                                                                         \
                |   grep -i -E 'm=|Duration|DRAM Throughput|Memory Throughput|L1/TEX Cache Throughput|L2 Cache Throughput|Compute'
            done
        done
    done
done
