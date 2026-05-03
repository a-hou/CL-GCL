
from torch_geometric.datasets import Amazon, Planetoid
import torch_geometric.transforms as T
from torch_sparse import SparseTensor, cat
import torch.utils.data
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, remove_self_loops, remove_isolated_nodes
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import List, Optional, Tuple, Union
import torch
import numpy as np
from torch_geometric.datasets import Planetoid, CitationFull
import scipy.sparse as sp
from torch_geometric.utils import to_undirected, subgraph, dense_to_sparse
from torch_geometric.utils import add_self_loops, degree, to_scipy_sparse_matrix
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon, DBLP
def get_sym(edge_index, self_loops = True, num_nodes: Optional[int] = None,
            edge_weight: Optional[torch.Tensor] = None):

    if self_loops==True:
        edge_index = add_self_loops(edge_index)[0]

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


def normalize_adj(edge_index, N):
    idx, val = get_sym(edge_index, self_loops=True, num_nodes=N)
    sparse_A_hat = torch.sparse.FloatTensor(idx, val, torch.Size([N, N]))
    return sparse_A_hat


def partition(edge_index, N, cluster):
    adj = SparseTensor(row=edge_index[0], col=edge_index[1],
        # value=torch.ones(E, device=data.edge_index.device),
        sparse_sizes=(N, N))
    recursive = False
    _, partptr_, perm_ = adj.partition(cluster, recursive)
    partptr = partptr_.tolist()
    perm = perm_.tolist()

    #修正分割结果，部分点会被单独且重复分为1类
    partptr = list(set(partptr))
    partptr.sort()

    return partptr, perm, len(partptr)-1

def create_community_matrix(N, num_communities, partptr, perm):
    # 创建一个大小为 (num_communities, N) 的零矩阵
    community_matrix = torch.zeros(num_communities, N, dtype=torch.float32)

    # 根据 partptr 和 perm 填充矩阵
    for i in range(num_communities):
        # 找到属于社区 i 的所有节点
        community_nodes = perm[partptr[i]:partptr[i + 1]]
        community_matrix[i, community_nodes] = 1

    return community_matrix

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index



def load_adj_neg(num_nodes, sample):


    row = np.repeat(range(num_nodes), sample)
    col = np.random.randint(0, num_nodes, size=num_nodes * sample)
    new_col = np.concatenate((col, row), axis=0)
    new_row = np.concatenate((row, col), axis=0)
    data = np.ones(new_col.shape[0])
    adj_neg = sp.coo_matrix((data, (new_row, new_col)), shape=(num_nodes, num_nodes))
    adj = np.array(adj_neg.sum(1)).flatten()
    adj_neg = sp.diags(adj) - adj_neg

    return adj_neg.toarray()


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def load_dataset(dataset_str):
    if dataset_str == 'cora' or dataset_str == 'pubmed' or dataset_str == 'citeseer':
        dataset = Planetoid(root='./dataset', name=dataset_str)
        data = dataset[0]
    elif dataset_str == 'Amazon-Photo':
        dataset = Amazon(root='./dataset', name='photo')
        data = dataset[0]
    elif  dataset_str == 'Coauthor-CS':
        dataset = Coauthor(root='./dataset', name='cs')
        data = dataset[0]
    elif dataset_str == 'WikiCS':
        # 加载 WikiCS 数据集
        dataset = WikiCS(root='./dataset')
        data = dataset[0]
    elif dataset_str == 'Amazon-Computers':
        # 加载 Amazon Computers 数据集
        dataset = Amazon(root='./dataset', name='computers')
        data = dataset[0]   
    elif dataset_str == 'WikiCS':
        # 加载 WikiCS 数据集
        dataset = WikiCS(root='./dataset')
        data = dataset[0]    
    return data



def build_community_adjacency(edge_index, num_nodes, partptr, perm, num_communities):
    """
    根据METIS划分结果生成社区间的邻接矩阵
    Args:
        edge_index: 原始图的边索引 [2, E]
        num_nodes: 原始图的节点数 (N)
        partptr: 划分的社区指针列表 (长度=社区数+1)
        perm: 节点重排列顺序
        num_communities: 社区数量 (len(partptr)-1)
    Returns:
        comm_adj: 社区邻接矩阵 [num_communities, num_communities] (torch.Tensor)
    """
    # 1. 将节点映射到其所属社区
    node_to_community = torch.zeros(num_nodes, dtype=torch.long)
    for comm_id in range(num_communities):
        start, end = partptr[comm_id], partptr[comm_id+1]
        node_to_community[perm[start:end]] = comm_id

    # 2. 转换原始边为社区间连接
    row, col = edge_index
    comm_row = node_to_community[row]  # 边的源节点所属社区
    comm_col = node_to_community[col]  # 边的目标节点所属社区

    # 3. 构建社区邻接矩阵（如果社区i和j之间有边则置1）
    comm_adj = torch.zeros((num_communities, num_communities), dtype=torch.float)
    comm_adj[comm_row, comm_col] = 1  # 记录所有社区间连接
    comm_adj = (comm_adj + comm_adj.t()).clamp(max=1)  # 确保对称性（无向图）

    return comm_adj


def community_adj_to_edge_index(community_adj):
    """
    将社区邻接矩阵转换为边索引格式（与 PyG 的 data.edge_index 一致）。
    
    Args:
        community_adj: 社区邻接矩阵 [num_communities, num_communities]，1 表示连接，0 表示无连接。
    
    Returns:
        edge_index: 边索引张量 [2, num_edges]，dtype=torch.long。
    """
    # 1. 获取所有非零元素的索引（即社区间的边）
    src, dst = torch.where(community_adj > 0)
    
    # 2. 移除自环（可选，根据需求决定是否保留）
    mask = src != dst
    src, dst = src[mask], dst[mask]
    
    # 3. 合并为 edge_index 格式 [2, num_edges]
    edge_index = torch.stack([src, dst], dim=0).long()
    
    # 4. 去重（如果是无向图，避免重复存储 i->j 和 j->i）
    edge_index = torch.unique(edge_index, dim=1)
    
    return edge_index

def random_planetoid_splits(num_classes, y, train_num, seed):
    # Set new random planetoid splits:
    # *  train_num * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    np.random.seed(seed)
    indices = []

    for i in range(num_classes):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:train_num] for i in indices], dim=0)

    rest_index = torch.cat([i[train_num:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    val_index = rest_index[:500]
    test_index = rest_index[500:1500]

    return train_index, val_index, test_index


def get_train_data(labels, tr_num, val_num, seed):
    np.random.seed(seed)
    labels_vec = labels.argmax(1)
    labels_num = labels_vec.max() + 1

    idx_train = []
    idx_val = []
    for label_idx in range(labels_num):
        pos0 = np.argwhere(labels_vec == label_idx).flatten()
        pos0 = np.random.permutation(pos0)
        idx_train.append(pos0[0:tr_num])
        idx_val.append(pos0[tr_num:val_num + tr_num])

    idx_train = np.array(idx_train).flatten()
    idx_val = np.array(idx_val).flatten()
    idx_test = np.setdiff1d(range(labels.shape[0]), np.union1d(idx_train, idx_val))

    idx_train = torch.LongTensor(np.random.permutation(idx_train))
    idx_val = torch.LongTensor(np.random.permutation(idx_val))
    idx_test = torch.LongTensor(np.random.permutation(idx_test))

    return idx_train, idx_val, idx_test


import numpy as np
import torch
import scipy.io as sio

def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]
def load_network_data(name):
    net = sio.loadmat('./data1/' + name + '.mat')
    X, A, Y = net['attrb'], net['network'], net['group']
    if name in ['cs', 'photo','computers','WikiCS']:
        Y = Y.flatten()
        Y = one_hot_encode(Y, Y.max() + 1).astype(int)
    return A, X, Y



