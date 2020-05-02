import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


if __name__ == "__main__":
	G = nx.read_gpickle("../reference.gpickle")
	print(G.number_of_nodes())
	print(G.number_of_edges())
	for node in G.nodes(data=True):
		print(node)
		break
	for edge in G.edges(data=True):
		print(edge)
		break
