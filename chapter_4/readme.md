To run profiling, first refers to the command within `run_container.sh` to start Nsight Compute environment, then refers to commands within `run_profile.sh` to profile each design and implementation

For example, to profile `SpMV-LB`:

1. Build binary

```bash
cd spmv_o2
mkdir build && cd build
cmake .. && make -j
```

2. To obtain overall statistics

```bash
ncu --target-processes all ../bin/spmv_csr  -m 8192 -n 65536 -r 0.5 -d 1024
```

3. To obtain transaction-specific statistics

```bash
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio ../bin/spmv_csr  -m 8192 -n 65536 -r 0.5 -d 1024
```
