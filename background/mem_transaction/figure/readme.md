# Figure Generator

## Usage

1. Create and config environment

```bash
# create
conda create --prefix=./env python=3.10

# install dependencies
conda install matplotlib
pip3 install brokenaxes
./env/bin/pip3 install mplfonts -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. Draw line figrue

```bash
python3 ./draw_line.py                                                                      \
    ../log/Tesla\ V100-PCIE-32GB/understand_transaction_l1_ld_overall_aligned_1.txt         \
    ../log/Tesla\ V100-PCIE-32GB/understand_transaction_l1_ld_partial_aligned_1.txt         \
    ../log/Tesla\ V100-PCIE-32GB/understand_transaction_l1_ld_overall_misaligned_1.txt      \
    ../log/Tesla\ V100-PCIE-32GB/understand_transaction_l2_ld_overall_aligned_1.txt         \
    ../log/Tesla\ V100-PCIE-32GB/understand_transaction_l2_ld_overall_misaligned_1.txt      \
    ../log/Tesla\ V100-PCIE-32GB/understand_transaction_dram_ld_overall_aligned_1.txt       \
    ../log/Tesla\ V100-PCIE-32GB/understand_transaction_dram_ld_overall_misaligned_1.txt 
```