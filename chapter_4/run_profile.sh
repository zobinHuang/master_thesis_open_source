declare -a m_array=(1024 2048 4096 8192 16384 32768 65536)
declare -a n_array=(65536)
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
                echo ""
                ncu --target-processes all ../bin/spmv_csr  \
                    -m $m -n $n -r $r -d $d                 \
                    |   grep -E 'm=|Duration|SOL L1/TEX Cache|SOL L2 Cache|Memory|SOL DRAM|SM'
                ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio ../bin/spmv_csr -m $m -n $n -r $r -d $d
            done
        done
    done
done
