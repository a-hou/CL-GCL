import torch
import argparse
import numpy as np
from model import CombinedModel
import os
import torch
from torch import Tensor
from torch_scatter import scatter
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from utils import load_adj_neg, load_dataset, normalize_adj,create_community_matrix,load_network_data
from utils import partition,random_planetoid_splits,get_train_data,build_community_adjacency,community_adj_to_edge_index
from torch_geometric.loader import ClusterData
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
import scipy.sparse as sp
import torch_geometric.transforms as T
from loss import contrastive_loss_with_community
import time
import process
from weight import get_community_weights,convert_to_community_weight_matrix
import warnings
# 禁用所有 FutureWarning 警告
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='citeseer',
                    help='dataset')
parser.add_argument('--seed', type=int, default=1,
                    help='seed')
parser.add_argument("--hidden", type=int, default=500,
                    help="layer hidden")
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument("--device", type=int, default=3, 
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument('--weight_decay', type=float, default=2e-5,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=40,
                    help='maximum number of epochs')
parser.add_argument('--tau', type=float, default=1,
                    help='    ')
parser.add_argument('--cluster', type=int, default=300,
                    help='    ')
parser.add_argument("--dropout", type=float, default=0.4,
                    help="dropout")
parser.add_argument("--epsilon", type=float, default=10,
                    help="epsilon")
parser.add_argument("--best_model_path", type=str, default='best_model.pkl', 
                    help="Path to save the best model")
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

data = load_dataset(args.dataset)
N, E, num_features = data.num_nodes, data.num_edges,  data.num_features
args.num_nodes = N
adj = normalize_adj(data.edge_index, N)
# 获取标签 (直接是整数类别)
lab = data.y.numpy()  # shape: (2708,)
edge_index = data.edge_index
# 转换为one-hot编码的Y (2708×7)
num_classes = len(torch.unique(data.y))  # 直接计算唯一标签数量
Y = torch.zeros(data.num_nodes, num_classes)
Y[torch.arange(data.num_nodes), data.y] = 1
Y = Y.numpy()  # shape: (2708, 7)
F_ori = data.x.to(device)

partptr, perm, args.cluster = partition(data.edge_index, N, cluster=args.cluster)

community_matrix = create_community_matrix(N, args.cluster, partptr, perm).to(device)

view1 = torch.zeros([args.cluster, num_features])
for j in range(args.cluster):
    view1[j, :] = torch.mean(F_ori[perm[partptr[j]:partptr[j + 1]], :], dim=0)
view1 = view1.to(device)

comm_adj = build_community_adjacency(edge_index, data.num_nodes, partptr, perm, args.cluster)
community_edge_index = community_adj_to_edge_index(comm_adj).to(device)
community_weights = get_community_weights(F_ori,view1,partptr, perm, device, epsilon=args.epsilon)
weight_matrix = convert_to_community_weight_matrix(community_weights, partptr, perm, N, device)
neg_weight_matrix = torch.ones(args.cluster, args.cluster, device=device) - torch.eye(args.cluster, device=device)
neg_weight_matrix = neg_weight_matrix / (args.cluster - 1)

model = CombinedModel(num_features, args.hidden, args.hidden, dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
data = data.to(device)

best_loss = 1e9
cnt_wait = 0
for epoch in range(1, args.epochs+1):
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    # start_time = time.time()
    model.train()
    optimizer.zero_grad()
    z_a = model(view1,community_edge_index)
    z = model(F_ori,data.edge_index)
    loss = contrastive_loss_with_community(z, community_matrix, z_a, weight_matrix, neg_weight_matrix, tau=args.tau, epsilon=args.epsilon)
    loss.backward()
    optimizer.step()
    # 打印日志，确保模型正在保存
    # 同步CUDA设备并记录结束时间
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    # end_time = time.time()
    # epoch_time = end_time - start_time
    # print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Time: {epoch_time:.4f}s')     
    if epoch % 20 == 0 or epoch == 1:
        #model.load_state_dict(torch.load(args.best_model_path))
        model.eval()
        with torch.no_grad():
            emb = model(data.x, data.edge_index)
            #torch.save(emb, r"E:\WeiXinkai\code\task3\task3-node_classifcation\node_embeddings_CL-GCL_citeseer.pt")
            #classify(args.dataset, emb.cpu(), data.y.cpu(), 50)
            embeds = emb.detach().cpu()
            Accuaracy_test_allK = []
            numRandom = 20

            for train_num in [20]:

                AccuaracyAll = []
                for random_state in range(numRandom):
                    # print(
                    #     "\n=============================%d-th random split with training num %d============================="
                    #     % (random_state + 1, train_num))

                    if train_num == 20:
                        if args.dataset in ['cora', 'citeseer', 'pubmed']:
                            # train_num per class: 20, val_num: 500, test: 1000
                            val_num = 500
                            idx_train, idx_val, idx_test = random_planetoid_splits(Y.shape[1], torch.tensor(lab), train_num,
                                                                                    random_state)
                        else:
                            # Coauthor CS, Amazon Computers, Amazon Photo
                            # train_num per class: 20, val_num per class: 30, test: rest
                            val_num = 30
                            idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

                    else:
                        val_num = 0  # do not use a validation set when the training labels are extremely limited
                        idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

                    train_embs = embeds[idx_train, :]
                    val_embs = embeds[idx_val, :]
                    test_embs = embeds[idx_test, :]

                    if train_num == 20:
                        # find the best parameter C using validation set
                        best_val_score = 0.0
                        for param in [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]:
                            LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, C=param)
                            LR.fit(train_embs, lab[idx_train])
                            val_score = LR.score(val_embs, lab[idx_val])
                            if val_score > best_val_score:
                                best_val_score = val_score
                                best_parameters = {'C': param}

                        LR_best = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, **best_parameters)

                        LR_best.fit(train_embs, lab[idx_train])
                        y_pred_test = LR_best.predict(test_embs)  # pred label
                        # print("Best accuracy on validation set:{:.4f}".format(best_val_score))
                        # print("Best parameters:{}".format(best_parameters))

                    else:  # not use a validation set when the training labels are extremely limited
                        LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0)
                        LR.fit(train_embs, lab[idx_train])
                        y_pred_test = LR.predict(test_embs)  # pred label

                    test_acc = accuracy_score(lab[idx_test], y_pred_test)
                    # print("test accuaray:{:.4f}".format(test_acc))
                    AccuaracyAll.append(test_acc)

                average_acc = np.mean(AccuaracyAll) * 100
                std_acc = np.std(AccuaracyAll) * 100
                print('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
                    numRandom, average_acc, std_acc, train_num, val_num))
                Accuaracy_test_allK.append(average_acc)


