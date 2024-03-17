## Understanding Memory Transaction

To build

```bash
mkdir build && cd build
cmake ..
make -j
```

To run

```bash
../bin/memory_testsuite
```

To get figure

```bash
conda create --prefix=./env python=3.10
conda activate ./env

conda install matplotlib
pip3 install brokenaxes
./env/bin/pip3 install mplfonts -i https://pypi.tuna.tsinghua.edu.cn/simple

cp ../figure/pkg/brokenaxes.py ./env/lib/python3.10/site-packages/brokenaxes.py

python3 ../figure/draw_line.py                                                          \
    ./log/Tesla\ V100-PCIE-32GB/understand_transaction_l1_ld_overall_aligned_1.txt      \
    ./log/Tesla\ V100-PCIE-32GB/understand_transaction_l1_ld_partial_aligned_1.txt      \
    ./log/Tesla\ V100-PCIE-32GB/understand_transaction_l1_ld_overall_misaligned_1.txt   \
    ./log/Tesla\ V100-PCIE-32GB/understand_transaction_l2_ld_overall_aligned_1.txt      \
    ./log/Tesla\ V100-PCIE-32GB/understand_transaction_l2_ld_overall_misaligned_1.txt   \
    ./log/Tesla\ V100-PCIE-32GB/understand_transaction_dram_ld_overall_aligned_1.txt    \
    ./log/Tesla\ V100-PCIE-32GB/understand_transaction_dram_ld_overall_misaligned_1.txt 
```
