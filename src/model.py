import copy
import math
import time
import argparse
import numpy as np
from sklearn.metrics import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn

def arg_parse():
    parser = argparse.ArgumentParser(description='Link prediction.')
    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--mode', type=str,
                        help='Link prediction mode. Disjoint or all.')
    parser.add_argument('--model', type=str,
                        help='GCN, GAT, GraphSAGE etc.')
    parser.add_argument('--edge_message_ratio', type=float,
                        help='Ratio of edges used for message-passing (only in disjoint mode).')
    parser.add_argument('--hidden_dim', type=int,
                        help='Hidden dimension of GNN.')
    parser.add_argument('--num_layers', type=int,
                        help='Number of conv layers.')

    parser.set_defaults(
            device='cuda:0',
            epochs=500,
            mode='all',
            model='GCN',
            edge_message_ratio=0.6,
            hidden_dim=16,
            num_layers=2,
    )
    return parser.parse_args()

def main():
    args = arg_parse()

if __name__ == '__main__':
    main()
