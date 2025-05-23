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
        import matplotlib.pyplot as plt
        plt.hist(pe.cpu().numpy(), bins=100)
        plt.title("Edge Sampling Probabilities")
        plt.savefig("edge_prob_histogram0.png", dpi=300)
    elif heuristic==1:
        print('Sparsify the graph by LOG of (1 + 2-hop degrees)')
        # Create adj_scipy from CPU tensors src, dst
        adj_scipy = to_scipy_sparse_matrix(torch.stack([src,dst]), num_nodes=N)
        d_u_squared = get_d_u_squared_matrix_ops(adj_scipy) # Returns float tensor, on CPU
        #cap = torch.quantile(d_u_squared, 0.6)
        #d_u_squared = torch.clamp(d_u_squared, max=cap)
        # Apply log1p transformation: log(1 + count)
        log_transformed_d_u_sq = torch.log1p(d_u_squared) # Input is float, output is float
        node_degree = degree(dst, data.num_nodes)

        # Calculate inverse terms for probability. Add epsilon to prevent 1/0.
        di_log_2hop = torch.nan_to_num(1. / (node_degree[src] + log_transformed_d_u_sq[src]))
        dj_log_2hop = torch.nan_to_num(1. / (node_degree[dst] + log_transformed_d_u_sq[dst]))
        pe = (di_log_2hop + dj_log_2hop).double()
        pe = pe / torch.sum(pe)
        import matplotlib.pyplot as plt
        plt.hist(pe.cpu().numpy(), bins=100)
        plt.title("Edge Sampling Probabilities")
        plt.savefig("edge_prob_histogram1.png", dpi=300)
    elif heuristic == 2:
        print('sparsify the graph by Jaccard similarity (common neighbors)')

        # Step 2.1: Build adjacency sets
        from collections import defaultdict
        adj = defaultdict(set)
        src_np, dst_np = src.numpy(), dst.numpy()
        for u, v in zip(src_np, dst_np):
            adj[u].add(v)
            if is_undirected:
                adj[v].add(u)

        # Step 2.2: Compute Jaccard similarity per edge
        inter = []
        union = []
        for u, v in zip(src_np, dst_np):
            inter.append(len(adj[u] & adj[v]))
            union.append(len(adj[u] | adj[v]))
            #sim = inter / union if union != 0 else 0.0
            #jaccard_scores.append(sim)

        # Convert to torch tensor
        #jaccard_scores = torch.tensor(jaccard_scores, dtype=torch.float64)
        #jaccard_scores = torch.log1p(jaccard_scores)
        # Invert for sparsification: low Jaccard = important edge
        #pe = 1.0 - jaccard_scores + 1e-6  # avoid zero prob
        inter = torch.tensor(inter, dtype=torch.float64)
        union = torch.tensor(union, dtype=torch.float64)
        d_e = torch.nan_to_num(1./(inter + union ))
        
        pe = (d_e).double()
        #pe = torch.nan_to_num(1./(jaccard_scores + 1e-6))
        pe = pe / pe.sum()
        import matplotlib.pyplot as plt
        plt.hist(pe.cpu().numpy(), bins=100)
        plt.title("Edge Sampling Probabilities")
        plt.savefig("edge_prob_histogram2.png", dpi=300)
    elif heuristic == 3:
        print('sparsify the graph by degree and  Jaccard similarity (common neighbors)')

        # Step 2.1: Build adjacency sets
        from collections import defaultdict
        adj = defaultdict(set)
        src_np, dst_np = src.numpy(), dst.numpy()
        for u, v in zip(src_np, dst_np):
            adj[u].add(v)
            if is_undirected:
                adj[v].add(u)

        # Step 2.2: Compute Jaccard similarity per edge
        inter = []
        union = []
        for u, v in zip(src_np, dst_np):
            inter.append(len(adj[u] & adj[v]))
            union.append(len(adj[u] | adj[v]))
            #sim = inter / union if union != 0 else 0.0
            #jaccard_scores.append(sim)

        # Convert to torch tensor
        #jaccard_scores = torch.tensor(jaccard_scores, dtype=torch.float64)
        #jaccard_scores = torch.log1p(jaccard_scores)
        # Invert for sparsification: low Jaccard = important edge
        #pe = 1.0 - jaccard_scores + 1e-6  # avoid zero prob
        
        inter = torch.tensor(inter, dtype=torch.float64)
        union = torch.tensor(union, dtype=torch.float64)
        jaccard = inter / union

        # Step 3: Get degrees
        node_degrees = degree(dst, data.num_nodes)  # torch_geometric.utils.degree
        deg_src = node_degrees[src]
        deg_dst = node_degrees[dst]
        deg_sum = (1. / deg_src) + (1. / deg_dst)

        # Step 4: Final sampling score
        eps = 1e-6
        pe = (deg_sum * (1. / (jaccard + 1))).double()
        pe = torch.nan_to_num(pe)  # avoid NaNs if any
        
        
        #pe = torch.nan_to_num(1./(jaccard_scores + 1e-6))
        pe = pe / pe.sum()
        import matplotlib.pyplot as plt
        plt.hist(pe.cpu().numpy(), bins=100)
        plt.title("Edge Sampling Probabilities")
        plt.savefig("edge_prob_histogram2.png", dpi=300)
    else:
      print("Select a heuristic:0,1,2")

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