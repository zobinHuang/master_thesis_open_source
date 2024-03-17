bin_name=memory_bandwidth
arch=70

cuobjdump ../bin/$bin_name -xelf all
cuobjdump -elf $bin_name.sm_$arch.cubin > $bin_name.sm_${arch}_sass.txt