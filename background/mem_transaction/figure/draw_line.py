import sys
import os
import json
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from mplfonts.bin.cli import init
from mplfonts import use_font
import argparse
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

# colors:
# https://matplotlib.org/stable/gallery/color/named_colors.html

# init of mplfonts.bin.cli
init()

plt.rcParams['font.family'] = 'SimHei'

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('axes', labelsize=15)

def draw_v100_l1_cache_global(param_index:int, cache_level:int):
    v100_data = open(sys.argv[param_index], 'r')

    # create figure instance
    fig, ax = plt.subplots(dpi=300)

    # fig = plt.figure()
    # ax = brokenaxes(
    #     ylims=((0, 130), (290, 305)), #设置y轴裂口范围
    #     hspace=0.05,#y轴裂口宽度             
    #     despine=False,#是否y轴只显示一个裂口
    #     diag_color='r',#裂口斜线颜色
    # )

    plt.grid()

    # ===== process v100 data =====

    lines = v100_data.readlines()

    num_elements = []
    byte_size = []
    gld_transactions = []
    global_load_requests = []
    gld_throughput = []
    gld_transactions_per_request = []
    l2_global_load_bytes = []
    l2_global_read_transactions = []
    l2_read_transactions = []
    l2_read_throughput = []
    dram_read_transactions = []
    dram_read_bytes = []
    dram_read_throughput = []

    text_rotation_angel = 0
    text_line_margin = 3

    value_map = dict()
    for id, line in enumerate(lines):
        statistics = line.split(" ")

        # makeup map
        if id == 0:
            for id, name in enumerate(statistics):
                value_map[name] = id
            continue

        def MAP(name:str):
            return statistics[value_map[name]]

        num_elements.append(float(MAP("num_elements")))
        byte_size.append(int(MAP("num_elements")) * 4)
        gld_transactions.append(float(MAP("gld_transactions")))
        global_load_requests.append(float(MAP("global_load_requests")))
        gld_throughput.append(float(MAP("gld_throughput")))
        gld_transactions_per_request.append(float(MAP("gld_transactions_per_request")))
        l2_global_load_bytes.append(float(MAP("l2_global_load_bytes")))
        l2_global_read_transactions.append(float(MAP("l2_global_load_bytes")) / 32)
        l2_read_transactions.append(float(MAP("l2_read_transactions")))
        l2_read_throughput.append(float(MAP("l2_read_throughput")))
        dram_read_transactions.append(float(MAP("dram_read_transactions")))
        dram_read_bytes.append(float(MAP("dram_read_bytes")))
        dram_read_throughput.append(float(MAP("dram_read_throughput")))

    # # theory l1 / l2 transaction size
    # ax.plot(
    #     [4, 4],
    #     [0, 8],
    #     '{}{}'.format("o", "--"),
    #     color=mcolors.CSS4_COLORS['black'],
    #     linewidth=2,
    #     ms = 0,
    #     mfc = mcolors.CSS4_COLORS['black'],
    #     mec = mcolors.CSS4_COLORS['black'],
    # )

    if cache_level == 1:
        # theory l1 size
        # ax.text(12.8, 1, 'L1 Cache (128 KB)', fontsize=8, color=mcolors.CSS4_COLORS['black'])
        ax.plot(
            [len(byte_size), len(byte_size)],
            [0, 33000],
            '{}{}'.format("o", "--"),
            color=mcolors.CSS4_COLORS['black'],
            linewidth=2,
            ms = 0,
            mfc = mcolors.CSS4_COLORS['black'],
            mec = mcolors.CSS4_COLORS['black'],
        )
    elif cache_level == 2:
        # theory l1 size
        ax.plot(
            [2, 2],
            [0, 33000],
            '{}{}'.format("o", "--"),
            color=mcolors.CSS4_COLORS['black'],
            linewidth=2,
            ms = 0,
            mfc = mcolors.CSS4_COLORS['black'],
            mec = mcolors.CSS4_COLORS['black'],
        )

        # theory l2 size
        ax.plot(
            [len(byte_size)-1.5, len(byte_size)-1.5],
            [0, 4200000],
            '{}{}'.format("o", "--"),
            color=mcolors.CSS4_COLORS['black'],
            linewidth=2,
            ms = 0,
            mfc = mcolors.CSS4_COLORS['black'],
            mec = mcolors.CSS4_COLORS['black'],
        )
    else: # cache_level == 3
        # theory l2 size
        ax.plot(
            [3.5, 3.5],
            [0, 4200000],
            '{}{}'.format("o", "--"),
            color=mcolors.CSS4_COLORS['black'],
            linewidth=2,
            ms = 0,
            mfc = mcolors.CSS4_COLORS['black'],
            mec = mcolors.CSS4_COLORS['black'],
        )

    # l1 request
    ax.plot(
        range(1,len(byte_size)+1), 
        global_load_requests,
        '{}{}'.format("o", "-"),
        color=mcolors.CSS4_COLORS['gray'],
        ms = 4,
        mfc=mcolors.CSS4_COLORS['gray'],
        mec=mcolors.CSS4_COLORS['gray'],
        label = "总内存请求数",
        # label="global_load_requests"
    )
    for x, y in zip(range(1,len(byte_size)+1), global_load_requests):
        ax.text(x, y+text_line_margin, int(y), ha='center', va='top', fontsize=6, rotation=text_rotation_angel, color=mcolors.CSS4_COLORS['gray'])

    # l1 transaction
    ax.plot(
        range(1,len(byte_size)+1), 
        gld_transactions,
        '{}{}'.format("o", "-"),
        color=mcolors.CSS4_COLORS['red'],
        ms = 4,
        mfc=mcolors.CSS4_COLORS['red'],
        mec=mcolors.CSS4_COLORS['red'],
        label = "总内存事务数",
        # label="gld_transactions"
    )
    for x, y in zip(range(1,len(byte_size)+1), gld_transactions):
        ax.text(x, y+text_line_margin, int(y), ha='center', va='bottom', fontsize=6, rotation=text_rotation_angel, color=mcolors.CSS4_COLORS['red'])

    # dram transaction
    ax.plot(
        range(1,len(byte_size)+1), 
        dram_read_transactions,
        '{}{}'.format("o", "-"),
        color=mcolors.CSS4_COLORS['darkorange'],
        ms = 4,
        mfc = mcolors.CSS4_COLORS['darkorange'],
        mec = mcolors.CSS4_COLORS['darkorange'],
        label = "Device Memory 事务",
        # label="dram_read_transactions"
    )
    for x, y in zip(range(1,len(byte_size)+1), dram_read_transactions):
        ax.text(x, y+text_line_margin, int(y), ha='center', va='bottom', fontsize=6, rotation=text_rotation_angel, color=mcolors.CSS4_COLORS['darkorange'])

    # l2 global read transaction
    ax.plot(
        range(1,len(byte_size)+1), 
        l2_global_read_transactions,
        '{}{}'.format("o", "-"),
        color=mcolors.CSS4_COLORS['navy'],
        ms = 4,
        mfc=mcolors.CSS4_COLORS['navy'],
        mec=mcolors.CSS4_COLORS['navy'],
        label = "L2 Cache 事务",
        # label="l2_global_read_transactions"
    )
    for x, y in zip(range(1,len(byte_size)+1), l2_global_read_transactions):
        ax.text(x, y+text_line_margin, int(y), ha='center', va='top', fontsize=6, rotation=text_rotation_angel, color=mcolors.CSS4_COLORS['navy'])

    ax.legend()
    ax.set_aspect(1)

    # set y axis
    ax.set_yscale('log')
    ax.set_ylabel("内存事务/请求数")

    # set x axis
    plt.xticks(range(1,len(byte_size)+1), byte_size)
    plt.xticks(rotation=30)
    ax.set_xlabel("访存规模 / Byte")
    
    plt.savefig(f'./{sys.argv[param_index]}.pdf', bbox_inches='tight')

def draw_v100_l1_cache_partial(param_index:int):
    v100_data = open(sys.argv[param_index], 'r')

    fig = plt.figure()
    ax = brokenaxes(
        ylims=((230, 290), (900, 1160)), #设置y轴裂口范围
        hspace=0.05,#y轴裂口宽度             
        despine=False,#是否y轴只显示一个裂口
        diag_color='r',#裂口斜线颜色
        yscale="log"
    )

    # ===== process v100 data =====

    lines = v100_data.readlines()

    num_elements = []
    byte_size = []
    gld_transactions = []
    global_load_requests = []
    gld_throughput = []
    gld_transactions_per_request = []
    l2_read_transactions = []
    l2_read_throughput = []
    dram_read_transactions = []
    dram_read_bytes = []
    dram_read_throughput = []

    value_map = dict()
    for id, line in enumerate(lines):
        statistics = line.split(" ")

        # makeup map
        if id == 0:
            for id, name in enumerate(statistics):
                value_map[name] = id
            continue

        def MAP(name:str):
            return statistics[value_map[name]]

        num_elements.append(float(MAP("num_elements")))
        byte_size.append(int(MAP("num_elements")) * 4)
        gld_transactions.append(float(MAP("gld_transactions")))
        global_load_requests.append(float(MAP("global_load_requests")))
        gld_throughput.append(float(MAP("gld_throughput")))
        gld_transactions_per_request.append(float(MAP("gld_transactions_per_request")))
        l2_read_transactions.append(float(MAP("l2_read_transactions")))
        l2_read_throughput.append(float(MAP("l2_read_throughput")))
        dram_read_transactions.append(float(MAP("dram_read_transactions")))
        dram_read_bytes.append(float(MAP("dram_read_bytes")))
        dram_read_throughput.append(float(MAP("dram_read_throughput")))

    # l1 request
    l1_req = ax.plot(
        byte_size,
        # range(1,len(byte_size)+1), 
        global_load_requests,
        '{}{}'.format("o", "-"),
        color=mcolors.CSS4_COLORS['red'],
        linewidth=2,
        ms = 3,
        mfc=mcolors.CSS4_COLORS['red'],
        mec=mcolors.CSS4_COLORS['red'],
        label = "总内存请求数",
        # label="global_load_requests"
    )

    # l1 transactions
    l1_trans = ax.plot(
        byte_size,
        # range(1,len(byte_size)+1), 
        gld_transactions,
        '{}{}'.format("o", "-"),
        color=mcolors.CSS4_COLORS['navy'],
        linewidth=2,
        ms = 3,
        mfc=mcolors.CSS4_COLORS['navy'],
        mec=mcolors.CSS4_COLORS['navy'],
        label = "总内存事务数",
        # label="gld_transactions"
    )

    ax.legend(loc='upper left')

    for id, byte in enumerate(byte_size):
        if byte % 128 == 0:
            ax.plot(
                [byte, byte],
                [0, global_load_requests[id]],
                '{}{}'.format("o", "--"),
                color=mcolors.CSS4_COLORS['black'],
                linewidth=2,
                ms = 0,
                mfc = mcolors.CSS4_COLORS['black'],
                mec = mcolors.CSS4_COLORS['black'],
            )

    def formatnum(x, pos):
        # return '%.1fe$^{6}$' % (x/1e6)
        return '%.1f' % (x/1e6)
    
    formatter = FuncFormatter(formatnum)

    for id, iax in enumerate(ax.get_axs()):
        iax.grid()
        # iax.set_xticks(np.arange(min(byte_size), max(byte_size), 128))
        iax.set_xticks(byte_size)
        
        if id == 0:
            iax.set_ylabel("内存事务/请求数")
            # iax.yaxis.set_major_formatter(formatter)
            # iax.set_yticks(np.arange(min(gld_transactions), max(gld_transactions), 2.8e5))
        elif id == 1:
            iax.set_xlabel("访问规模 / Byte")
            # iax.yaxis.set_major_formatter(formatter)
            iax.set_yticks(global_load_requests)
    
    fig.autofmt_xdate(rotation=90)

    plt.savefig(f'./{sys.argv[param_index]}.pdf', bbox_inches='tight')


if __name__ == '__main__':
   # l1 aligned load
   draw_v100_l1_cache_global(1, cache_level=1)
   draw_v100_l1_cache_partial(2)

   # l1 misaligned load
   draw_v100_l1_cache_global(3, cache_level=1)
   
   # l2 aligned load
   draw_v100_l1_cache_global(4, cache_level=2)

   # l2 misaligned load
   draw_v100_l1_cache_global(5, cache_level=2)

   # dram aligned load
   draw_v100_l1_cache_global(6, cache_level=3)

   # dram misaligned load
   draw_v100_l1_cache_global(7, cache_level=3)

