import pdb
import time
import random
random.seed(42)
import numpy as np
import torch
import scipy.sparse as sp
import dspar.cpp_extension.sampler as sampler
from torch_geometric.utils import degree, to_undirected,from_scipy_sparse_matrix, to_scipy_sparse_matrix


def get_d_u_squared_matrix_ops(adj_matrix_scipy):
    """
    Calculates d_u^(2) for all nodes using sparse matrix operations.
    d_u^(2) is the number of unique nodes within 2 hops of u, excluding u itself.

    Args:
        adj_matrix_scipy (sp.csr_matrix): The input adjacency matrix (scipy sparse).

    Returns:
        torch.Tensor: A tensor where the i-th element is d_i^(2).
    """
    num_nodes = adj_matrix_scipy.shape[0]

    adj_normalized = adj_matrix_scipy.copy()
    adj_normalized.data[:] = 1 # Ensure binary
    adj_normalized.setdiag(0)  # Remove self-loops from A part
    adj_normalized.eliminate_zeros()

    #adj_plus_identity = adj_normalized + sp.eye(num_nodes, format='csr')
    
    # (A) @ (A) gives reachability within 2 hops (paths of length 0, 1, 2)
    reach_leq_2_hops_matrix = (adj_normalized @ adj_normalized).astype(bool) # Binarize

    n2_u_counts_including_self = np.array(reach_leq_2_hops_matrix.sum(axis=1)).flatten()
    d_u_squared_np = n2_u_counts_including_self - 1
    
    # Ensure no negative degrees (e.g., for completely isolated nodes where N2(u) might be just {u})
    d_u_squared_np[d_u_squared_np < 0] = 0 

    return torch.from_numpy(d_u_squared_np).float() # Return as torch.Tensor
# 

def maybe_sparsfication(data, dataset, follow_by_subgraph_sampling,heuristic= 0, random=False, is_undirected=True, reweighted=True):
    N, E = data.num_nodes, data.num_edges
    src, dst = data.edge_index
    if dataset == 'ogbn-arxiv':
        epsilon = 0.25 if not random else 0.35
    elif dataset == 'reddit2':
        epsilon = 0.3 if not random else 0.32
    elif dataset == 'ogbn-products':
        epsilon = 0.4 if not random else 0.45
    elif dataset == 'yelp':
        epsilon = 0.5 if not random else 0.6
    elif dataset == 'ogbn-proteins':
        epsilon = 0.25

    if follow_by_subgraph_sampling and dataset == 'ogbn-products':
        epsilon = 0.15 if not random else 0.2

    print(f'epsilon: {epsilon}')
    Q = int(0.16 * N * np.log(N) / epsilon ** 2)
    print(f"Q: {Q}")
    print(f'E/Q ratio: {E/Q}')
    print(f'E/nlogn ratio: {E/N/np.log(N)}')
    print('sparsify the input graph')
    data = data.clone()
    s = time.time()
    epsilon_for_log_degree_inverse = 1e-6 
    if random:
        pe = torch.ones(size=(E,), dtype=torch.double) / E
    elif heuristic==0:
        print('sparsify the graph by degrees')
        node_degree = degree(dst, data.num_nodes)
        di, dj = torch.nan_to_num(1. / node_degree[src]), torch.nan_to_num(1. / node_degree[dst])
        pe = (di + dj).double()
        pe = pe / torch.sum(pe)
    elif heuristic==1:
        print('Sparsify the graph by LOG of (1 + 2-hop degrees)')
        # Create adj_scipy from CPU tensors src, dst
        adj_scipy = to_scipy_sparse_matrix(torch.stack([src,dst]), num_nodes=N)
        d_u_squared = get_d_u_squared_matrix_ops(adj_scipy) # Returns float tensor, on CPU
        
        # Apply log1p transformation: log(1 + count)
        log_transformed_d_u_sq = torch.log(d_u_squared) # Input is float, output is float
        di, dj = torch.nan_to_num(1. / node_degree[src]), torch.nan_to_num(1. / node_degree[dst])

        # Calculate inverse terms for probability. Add epsilon to prevent 1/0.
        di_log_2hop = 1.0 / (di + log_transformed_d_u_sq[src] + epsilon_for_log_degree_inverse)
        dj_log_2hop = 1.0 / (dj+ log_transformed_d_u_sq[dst] + epsilon_for_log_degree_inverse)
        pe = (di_log_2hop + dj_log_2hop).double()
    p_cumsum = torch.cumsum(pe, 0)
    print(f'cal edge distribution used {time.time() - s} sec')
    # For reproducibility, we manually set the seed of graph sparsification to 42. We note that this seed is only effective for the graph sparsification, 
    # it does not impact any following process.
    seed_val = 42
    s = time.time()
    sampled = sampler.edge_sample(p_cumsum, Q, seed_val)
    print(f'sample edge used {time.time() - s} sec')
    e_indices, e_cnt = torch.unique(sampled, return_counts=True)
    new_graph = e_cnt / Q / pe[e_indices]
    new_src, new_dst = src[e_indices], dst[e_indices]
    edge_index = torch.cat([new_src.view(1, -1), new_dst.view(1, -1)], dim=0)
    edge_attr = new_graph.float()
    if is_undirected:
        data.edge_index, data.edge_attr = to_undirected(edge_index, edge_attr)
    else:
        data.edge_index, data.edge_attr = edge_index, edge_attr 
    if not reweighted:
        print('not reweight')
        data.edge_attr = None
    print(f'before sparsification, num_edges: {E}, after sparsification, num_edges: {new_src.shape[0]}, ratio: {new_src.shape[0] / E}')
    return data