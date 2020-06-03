import json
import torch
import numpy as np
import networkx as nx
import pickle as pkl

from os import listdir
from os.path import isfile, join

class Entity(object):
	def __init__(self, G, alpha=0.7, beta=20, rand_num=100):
		self.G = G
		self.entities = self.load_files()
		self.neighbors = self.hop_neighbors(G)
		self.edges = self.similarity_generation(alpha, beta)
		self.alpha = alpha
		num_edges = 0
		for edge in self.edges:
			num_edges += len(self.edges[edge])
		random_edges = self.random_generation(rand_num)
		print('Adding {} internal edges and {} random edges'.format(num_edges, random_edges))
		self.edge_index = self.generate_edge_index()
		assert self.edge_index.size(1) == (num_edges + random_edges) * 2

	def generate_edge_index(self):
		edges = []
		for node_id in self.edges:
			for neigh in self.edges[node_id]:
				edges.append([node_id, neigh])
		edge_index = torch.LongTensor(edges)
		edge_index = torch.cat([edge_index, torch.flip(edge_index, [1])], dim=0)
		return edge_index.permute(1, 0)

	def similarity_generation(self, alpha, beta):
		add_edges = {}
		for i in range(self.G.number_of_nodes()):
			if i >= 10533:
				break
			node_entity = self.entities[i]
			node_neighbors = self.neighbors[i]
			if len(node_neighbors) < beta:
				for neigh in node_neighbors:
					if neigh in self.entities:
						neigh_entity = self.entities[neigh]
						score = self.similarity_compare(node_entity, neigh_entity)
						if score > alpha:
							if i in add_edges:
								add_edges[i].append(neigh)
							else:
								add_edges[i] = [neigh]
		return add_edges

	def random_generation(self, rand_num):
		count = 0
		num_nodes = self.G.number_of_nodes()
		added = 0
		while count < rand_num:
			node_s = torch.randint(low=0, high=num_nodes, size=()).item()
			node_e = torch.randint(low=0, high=num_nodes, size=()).item()
			if node_s == node_e:
				continue
			if node_s in self.edges and node_e in self.edges[node_s]:
				continue
			if node_e in self.edges and node_s in self.edges[node_e]:
				continue
			if node_s not in self.entities or node_e not in self.entities:
				continue
			s_entity = self.entities[node_s]
			e_entity = self.entities[node_e]
			score = self.similarity_compare(s_entity, e_entity)
			if score > self.alpha:
				if node_s in self.edges:
					self.edges[node_s].append(node_e)
				else:
					self.edges[node_s] = [node_e]
				added += 1
			count += 1
		return added

	def similarity_compare(self, a, b):
		if len(a) == 0 or len(b) == 0:
			return 0
		return len(a.intersection(b)) / len(a.union(b))

	def hop_neighbors(self, G):
		neighbors = {}
		for node_id in self.entities:
			node_neighbors = set()
			one_hop_neighbors = set(G.neighbors(node_id))
			for one_neigh in one_hop_neighbors:
				two_hop_neighbors = set(G.neighbors(one_neigh))
				for two_neigh in two_hop_neighbors:
					if two_neigh not in one_hop_neighbors and two_neigh != node_id:
						node_neighbors.add(two_neigh)
			neighbors[node_id] = node_neighbors
		return neighbors

	def load_files(self):
		with open('../data/bern.pkl', 'rb') as f:
			return pkl.load(f)
