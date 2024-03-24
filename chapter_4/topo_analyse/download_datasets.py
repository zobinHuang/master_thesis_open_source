import os
import struct
from ogb.nodeproppred import NodePropPredDataset
import numpy as np
from scipy.sparse import csr_matrix

dataset_name = "ogbn-products"
dataset_cast_path = "./topo"
    
# TODO: draw some figure
def __analyse_topo(num_nodes:int, num_edges:int, csr_ptr:list, csr_col:list):
    inter_row_distribution = []

    print(f"#nodes: {num_nodes}, #edges: {num_edges}")
    print(f"global sparcity: {1-(num_edges / (num_nodes * num_nodes))}")

    for i in range(0, len(csr_ptr)-1):
        inter_row_distribution.append(csr_ptr[i+1] - csr_ptr[i])
    
    inter_np = np.array(inter_row_distribution)
    print(f"inter row distribution:"
          f"p10({np.percentile(inter_np, 10)}), p50({np.percentile(inter_np, 50)}), p99({np.percentile(inter_np, 99)}), "
          f"mean({np.mean(inter_np)}), min({np.min(inter_np)}), max({np.max(inter_np)}), "
          f"var({np.var(inter_np)})")

dataset = NodePropPredDataset(name = dataset_name)
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph, label = dataset[0]

edge_indices = graph['edge_index']
num_edges = len(edge_indices[0])
num_nodes =  graph['num_nodes']

# edge_indices = [
#     [0, 0, 1, 1, 2, 3, 3],
#     [0, 2, 1, 3, 2, 0, 3]
# ]
# num_edges = len(edge_indices[0])
# num_nodes = 4



if not os.path.exists(dataset_cast_path):
    os.makedirs(dataset_cast_path)

print("converting to csr...")
csr = csr_matrix(([1 for _ in range(0, num_edges)], (edge_indices[0], edge_indices[1])), shape=(num_nodes, num_nodes))

print("analyse topo...")
__analyse_topo(num_nodes, num_edges, csr.indptr, csr.indices)

if not os.path.exists(f"{dataset_cast_path}/{dataset_name}"):
    os.makedirs(f"{dataset_cast_path}/{dataset_name}")

    print("dumping csr file...")
    with open(f"{dataset_cast_path}/{dataset_name}/csr.bin", 'wb') as file:
        size_of_row_ptr = len(csr.indptr)
        # number of edges
        file.write(num_edges.to_bytes(8, byteorder='big', signed=False))
        # size of row_ptr
        file.write(size_of_row_ptr.to_bytes(8, byteorder='big', signed=False))
        # row_ptr
        for row_ptr in csr.indptr:
            file.write(row_ptr.item().to_bytes(8, byteorder='big', signed=False))
        # col
        for col in csr.indices:
            file.write(col.item().to_bytes(8, byteorder='big', signed=False))
        # value
        for data in csr.indptr:
            file.write(data.item().to_bytes(8, byteorder='big', signed=False))
        file.close()


    print("converting to coo...")
    coo = csr.tocoo()

    print("dumping coo file...")
    with open(f"{dataset_cast_path}/{dataset_name}/coo.bin", 'wb') as file:
        # number of edges
        file.write(num_edges.to_bytes(8, byteorder='big', signed=False))
        # row
        for s_vid in coo.row:
            file.write(s_vid.item().to_bytes(8, byteorder='big', signed=False))
        # col
        for d_vid in coo.col:
            file.write(d_vid.item().to_bytes(8, byteorder='big', signed=False))
        # value
        for data in coo.data:
            file.write(data.item().to_bytes(8, byteorder='big', signed=False))
        file.close()

    # print("converting to adj...")
    # adj_array = csr.toarray()

    # print("dumping adj file...")
    # with open(f"{dataset_cast_path}/{dataset_name}/adj.bin", 'wb') as file:
    #     # number of nodes
    #     file.write(num_nodes.to_bytes(8, byteorder='big', signed=False))
    #     # array
    #     for row in adj_array:
    #         for ele in row:
    #             file.write(ele.item().to_bytes(8, byteorder='big', signed=False))
    #     file.close()
