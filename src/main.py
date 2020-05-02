import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
	G = nx.read_gpickle("../data/reference.gpickle")
	print("Number of nodes:", G.number_of_nodes())
	print("Number of edges:",G.number_of_edges())
	for node in G.nodes(data=True):
		print(node)
		break
	for edge in G.edges(data=True):
		print(edge)
		break
