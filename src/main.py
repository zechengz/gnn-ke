import torch
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
	G_old = nx.read_gpickle("../data/reference_tensor_old.gpickle")
	print("Number of nodes:", G_old.number_of_nodes())
	print("Number of edges:",G_old.number_of_edges())
	print(G_old.nodes[0])
	G = nx.read_gpickle("../data/reference_tensor.gpickle")
	print("Number of nodes:", G.number_of_nodes())
	print("Number of edges:",G.number_of_edges())
	print(G.nodes[0])
	for i in range(G.number_of_nodes()):
		if not torch.all(G.nodes[i]['node_feature'].eq(G_old.nodes[i]['node_feature'])):
			print('unequal')
