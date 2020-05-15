import copy
import math
import time
import random
import argparse
import numpy as np
from sklearn.metrics import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn

import networkx as nx

def arg_parse():
    parser = argparse.ArgumentParser(description='Link prediction.')
    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--model', type=str,
                        help='GAT and GraphSAGE etc.')
    parser.add_argument('--hidden_dim', type=int,
                        help='Hidden dimension of GNN.')

    parser.set_defaults(
            device='cuda:0',
            epochs=200,
            model='SAGE',
            hidden_dim=16,
    )
    return parser.parse_args()

class Net(torch.nn.Module):
    def __init__(self, input_dim, num_classes, args):
        super(Net, self).__init__()
        self.model = args.model
        if self.model == 'GAT':
            self.conv1 = pyg_nn.GATConv(input_dim, args.hidden_dim)
            self.conv2 = pyg_nn.GATConv(args.hidden_dim, num_classes)
        elif self.model == 'SAGE':
            self.conv1 = pyg_nn.SAGEConv(input_dim, args.hidden_dim)
            self.conv2 = pyg_nn.SAGEConv(args.hidden_dim, num_classes)
        else:
            raise ValueError('unknown conv')
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, node_feature, edge_index, edge_label_index):
        x = F.dropout(node_feature, p=0.2, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)

        nodes_first = torch.index_select(x, 0, edge_label_index[0,:].long())
        nodes_second = torch.index_select(x, 0, edge_label_index[1,:].long())
        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        return pred

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

def train(model, dataloaders, optimizer, args, scheduler=None):
    val_max = -math.inf
    best_model = model
    for epoch in range(1, args.epochs):
        model.train()
        optimizer.zero_grad()

        node_feature = dataloaders['train']['node_feature'].to(args.device)
        edge_index = dataloaders['train']['edge_index'].to(args.device)
        edge_index_label = dataloaders['train']['edge_label_index'].to(args.device)
        labels = dataloaders['train']['link_label'].to(args.device)

        pred = model(node_feature, edge_index, edge_index_label)
        loss = model.loss(pred, labels.type(pred.dtype))
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        log = 'Epoch: {:03d}, Loss: {:.8f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        rocs, _, accs = test(model, dataloaders, args)

        print(log.format(epoch, loss.item(), rocs['train'], rocs['val'], rocs['test']))
        print(log.format(epoch, loss.item(), accs['train'], accs['val'], accs['test']))
        if val_max < accs['val']:
            val_max = accs['val']
            best_model = copy.deepcopy(model)

    log = 'Best, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    rocs, _, accs = test(best_model, dataloaders, args)
    print(log.format(rocs['train'], rocs['val'], rocs['test']))
    print(log.format(accs['train'], accs['val'], accs['test']))

def test(model, dataloaders, args):
    model.eval()
    rocs = {}
    losses = {}
    accs = {}

    for mode, dataloader in dataloaders.items():
        roc = 0
        loss = 0
        acc = 0
        pred = model(dataloader['node_feature'], dataloader['edge_index'], dataloader['edge_label_index'])
        loss += model.loss(pred, dataloader['link_label'].type(pred.dtype)).cpu().data.numpy()
        sig_res = torch.sigmoid(pred.flatten().data.cpu())
        sig_res = sig_res.numpy()
        pred_label = np.zeros_like(sig_res, dtype=np.int64)
        pred_label[np.where(sig_res > 0.5)[0]] = 1
        pred_label[np.where(sig_res <= 0.5)[0]] = 0
        acc = np.mean(pred_label == dataloader['link_label'].flatten().cpu().numpy())
        roc = roc_auc_score(dataloader['link_label'].flatten().cpu().numpy(), 
                            torch.sigmoid(pred).flatten().data.cpu().numpy())
        rocs[mode] = roc
        losses[mode] = loss
        accs[mode] = acc
    return rocs, losses, accs

def read_nx_graph():
    G = nx.read_gpickle("../data/reference_tensor.gpickle")
    edges = list(G.edges)
    random.shuffle(edges)
    edge_index = torch.LongTensor(list(edges)).permute(1, 0)
    node_feature = []
    node_feature_check_1 = None
    node_feature_check_2 = None
    for i, node in enumerate(G.nodes(data=True)):
        if i == 100:
            node_feature_check_1 = node[1]['node_feature']
        if i == 1000:
            node_feature_check_2 = node[1]['node_feature']
        node_feature.append(node[1]['node_feature'])
    node_feature = torch.stack(node_feature, dim=0)
    assert torch.all(node_feature_check_1 == node_feature[100]).item() == True
    assert torch.all(node_feature_check_2 == node_feature[1000]).item() == True
    return G, edge_index, node_feature

def split_edges(edge_index, num_edges, split_ratio=[0.85, 0.05, 0.1]):
    num_edges_train = 1 + int(split_ratio[0] * num_edges)
    num_edges_val = 1 + int(split_ratio[1] * num_edges)
    edges_train = edge_index[:, :num_edges_train]
    edges_val = edge_index[:, num_edges_train:num_edges_train + num_edges_val]
    edges_test = edge_index[:, num_edges_train + num_edges_val:]
    print("Total number of edges: {}".format(num_edges))
    print("Number of edges train {}, validation {}, test {}".format(edges_train.size(1), edges_val.size(1), edges_test.size(1)))
    assert edges_train.size(1) + edges_val.size(1) + edges_test.size(1) == num_edges
    return edges_train, edges_val, edges_test

def create_edges(edges_train, edges_val, edges_test, num_nodes):
    edges_train_neg = negative_sampling(edges_train, num_nodes, 
                                            num_neg_samples=edges_train.size(1))
    edges_val_neg = negative_sampling(torch.cat([edges_train, edges_val], dim=1), num_nodes, 
                                            num_neg_samples=edges_val.size(1))
    edges_test_neg = negative_sampling(torch.cat([edges_train, edges_val, edges_test], dim=1), num_nodes, 
                                            num_neg_samples=edges_test.size(1))
    edges_train_total = torch.cat([edges_train, edges_train_neg], dim=1)
    edges_val_total = torch.cat([edges_val, edges_val_neg], dim=1)
    edges_test_total = torch.cat([edges_test, edges_test_neg], dim=1)
    return edges_train_total, edges_val_total, edges_test_total

def create_labels(edges_train_total, edges_val_total, edges_test_total):
    train_size = edges_train_total.size(1) // 2
    val_size = edges_val_total.size(1) // 2
    test_size = edges_test_total.size(1) // 2
    train_labels = torch.cat([torch.ones([train_size,], dtype=torch.long), 
                             torch.zeros([train_size,],dtype=torch.long)])
    val_labels = torch.cat([torch.ones([val_size,], dtype=torch.long), 
                           torch.zeros([val_size,],dtype=torch.long)])
    test_labels = torch.cat([torch.ones([test_size,], dtype=torch.long), 
                            torch.zeros([test_size,],dtype=torch.long)])
    return train_labels, val_labels, test_labels

def main():
    args = arg_parse()
    G, edge_index, node_feature = read_nx_graph()
    edges_train, edges_val, edges_test = split_edges(edge_index, G.number_of_edges())
    edges_train_total, edges_val_total, edges_test_total = \
                        create_edges(edges_train, edges_val, edges_test, G.number_of_nodes())
    train_labels, val_labels, test_labels = \
                        create_labels(edges_train_total, edges_val_total, edges_test_total)
    input_dim = node_feature.size(1)
    num_classes = 2

    dataloaders = {}
    dataloaders['train'] = {}
    dataloaders['train']['node_feature'] = node_feature.to(args.device)
    dataloaders['train']['edge_index'] = edge_index.to(args.device)
    dataloaders['train']['edge_label_index'] = edges_train_total.to(args.device)
    dataloaders['train']['link_label'] = train_labels.to(args.device)
    dataloaders['val'] = {}
    dataloaders['val']['node_feature'] = node_feature.to(args.device)
    dataloaders['val']['edge_index'] = edge_index.to(args.device)
    dataloaders['val']['edge_label_index'] = edges_val_total.to(args.device)
    dataloaders['val']['link_label'] = val_labels.to(args.device)
    dataloaders['test'] = {}
    dataloaders['test']['node_feature'] = node_feature.to(args.device)
    dataloaders['test']['edge_index'] = edge_index.to(args.device)
    dataloaders['test']['edge_label_index'] = edges_test_total.to(args.device)
    dataloaders['test']['link_label'] = test_labels.to(args.device)

    model = Net(input_dim, num_classes, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train(model, dataloaders, optimizer, args, scheduler=scheduler)

if __name__ == '__main__':
    main()
