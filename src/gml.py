import networkx as nx

def write_gml():
	G = nx.read_gpickle("../data/reference.gpickle")
	nodes = list(G.nodes(data=False))
	edges = list(G.edges(data=False))
	G1 = nx.DiGraph()
	G1.add_nodes_from(nodes)
	G1.add_edges_from(edges)
	nx.write_gml(G1, "../data/reference.gml")

if __name__ == "__main__":
	write_gml()
