import torch
import torch.nn.functional as F

def get_community_weights(z, view1, partptr, perm, device, epsilon=1e-5):
    N = z.size(0)
    cluster_count = view1.size(0)
    community_weights = {}

    for j in range(cluster_count):
        start_idx = partptr[j]
        end_idx = partptr[j + 1]
        community_nodes = z[perm[start_idx:end_idx]]
        anchor_point = view1[j].unsqueeze(0)  # (1, F)
        
        weights = []
        for i in range(len(community_nodes)):
            x_i = community_nodes[i]  # (F,)
            x_anchor = anchor_point  # (1, F)

            G = x_anchor @ x_anchor.T  # (1, 1)
            b = x_anchor @ x_i.unsqueeze(-1)  # (1, 1)
            G = G.to(device)
            b = b.to(device)

            # 确保 G 可逆
            if torch.det(G + epsilon * torch.eye(1, device=device)) == 0:
                w = torch.ones(1, device=device)  # 退化解
            else:
                w = torch.linalg.solve(G + epsilon * torch.eye(1, device=device), b)  # (1, 1)
            
            # 确保 w 是标量（shape=[]）
            w = F.relu(w).squeeze()  # 从 (1, 1) -> []
            
            # 归一化
            if w.sum() > 0:
                w = (w / w.sum()).item()  # 转为 Python 标量
            else:
                w = 1.0  # 默认权重
            
            weights.append(w)

        # 将 weights 转为张量（确保形状一致）
        community_weights[j] = torch.tensor(weights, device=device)  # shape: [num_nodes_in_community]

    return community_weights


import torch
import torch.nn.functional as F

def convert_to_community_weight_matrix(community_weights, partptr, perm, N, device):
    num_communities = len(community_weights)
    weight_matrix = torch.zeros((N, num_communities), device=device)

    for j, weights in community_weights.items():
        start_idx = partptr[j]
        end_idx = partptr[j + 1]
        community_nodes_idx = perm[start_idx:end_idx]
        
        for i, node_idx in enumerate(community_nodes_idx):
            weight_matrix[node_idx, j] = weights[i]  # weights[i] 已经是标量

    return weight_matrix
