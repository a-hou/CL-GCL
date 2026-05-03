import torch
import torch.nn.functional as F
from weight import convert_to_community_weight_matrix

def contrastive_loss_with_community(z, community_matrix,view1, weight_matrix, neg_weight_matrix, tau, epsilon):
    """
    计算基于社区锚点的对比损失
    :param z: 节点特征矩阵 (N, F)
    :param community_weights: 每个社区的节点与锚点的权重字典
    :param partptr: 节点分区指针
    :param perm: 节点分区排列顺序
    :param N: 节点总数
    :param device: 设备信息
    :param view1: 每个社区的锚点 (num_communities, F)
    :param weight_matrix: 传入的社区加权矩阵 (N, num_communities)
    :param neg_weight_matrix: 传入的负样本加权矩阵 (num_communities, num_communities)
    :param tau: 温度参数
    :param epsilon: 避免除零的常数
    :return: 对比损失
    """
    z = F.normalize(z)
    view1 = F.normalize(view1)
    sim_matrix = torch.matmul(view1, z.T)  # (num_communities, N)

    # 使用社区矩阵选取每个节点对应的社区锚点相似度
    community_sim = community_matrix * sim_matrix  # (num_communities, N)

    # 4. 温度缩放并计算指数
    sim_matrix = torch.exp(community_sim / tau)  # ( num_communities,N)

    # 5. 正样本的相似度（社区锚点与该社区内所有节点之间的相似性）
    pos_sim = sim_matrix * weight_matrix.T  # 只考虑社区内部的节点对
    pos_sim_sum = pos_sim.sum(1) + epsilon  # 计算正样本的总和

    # 6. 负样本的相似度计算
    # 计算社区间的相似度矩阵
    anchor_dot_product = torch.matmul(view1, view1.T)  # (num_communities, num_communities)
    negative_sim = torch.exp(anchor_dot_product / tau)  # 每对社区之间的相似度，按温度缩放

    # 7. 计算负样本相似度矩阵：sim(zi, zk) * w_nik
    neg_sim = negative_sim * neg_weight_matrix  # 计算负样本的加权相似度矩阵

    # 8. 负样本的总和
    neg_sim_sum = neg_sim.sum(1) + epsilon  # 每个社区的负样本总和

    # 9. 计算对比损失
    loss = -torch.log(pos_sim_sum / (pos_sim_sum + neg_sim_sum)).mean()

    return loss
