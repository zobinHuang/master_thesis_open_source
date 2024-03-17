file_name=ptx_memory_bandwidth
# nvcc --std c++17 -g -G -Xcompiler -O0 -Xptxas -O0 -lineinfo -O0 --ptx ../src/$file_name.cu -o ../src/$file_name.o
nvcc --std c++17 --ptx ../src/$file_name.cu -o ../src/$file_name.o