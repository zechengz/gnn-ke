import json
import numpy as np
import networkx as nx
import pandas as pd

from os import listdir
from os.path import isfile, join

def load_clean_metadata():
	df = pd.read_csv("../data/metadata.csv")
	n = len(df)
	keys = ['sha', 'title', 'abstract']
	d = {'sha':[], 'title':[], 'abstract':[]}
	for i in range(n):
		row_data = df.iloc[i]
		if row_data['sha'] != '' and row_data['title'] != '' and row_data['abstract'] != '':
			d['sha'].append(row_data['sha'])
			d['title'].append(row_data['title'])
			d['abstract'].append(row_data['abstract'])
	df_save = pd.DataFrame.from_dict(d)
	df_save.to_csv("../data/metadata_clean.csv", index=False)

def construct_edge_list():
	df = pd.read_csv("../data/metadata_clean.csv")
	n = len(df)
	json_files = [f for f in listdir('../data') if isfile(join('../data', f))]
	json_files = set(json_files)
	edges = []
	nodes = set()

	for i in range(n):
		row_data = df.iloc[i]
		if type(row_data['sha']) == str and type(row_data['title']) == str:
			nodes.add(row_data['title'])

	for i in range(n):
		row_data = df.iloc[i]
		if type(row_data['sha']) != str or type(row_data['title']) != str:
			continue
		filename = row_data['sha'] + '.json'
		tital_head = row_data['title']
		tital_head = tital_head.replace("\t", " ")
		if filename not in json_files:
			continue
		with open("../data/" + filename, 'r') as f:
			data = json.load(f)
		if 'bib_entries' not in data or len(data['bib_entries']) == 0:
			continue
		for key in data['bib_entries']:
			if 'title' not in data['bib_entries'][key]:
				continue
			title_tail = data['bib_entries'][key]['title']
			title_tail = title_tail.replace("\t", " ")
			if type(title_tail) == str and title_tail in nodes:
				edges.append([tital_head, title_tail])
		print(i, len(edges))
	print("length", len(edges))
	save_edge_list_txt(edges)

def save_edge_list_txt(edges):
	with open("../data/edge_list.txt", 'w') as f:
		for i, edge in enumerate(edges):
			string = edge[0] + "\t" + edge[1] + "\n"
			f.write(string)

def read_edge_list_txt():
	edges = []
	with open("../data/edge_list.txt", 'r') as f:
		for i, line in enumerate(f):
			line = line[0:len(line) - 1]
			line = line.split("\t")
			edges.append([line[0], line[1]])
	return edges

def construct_nx_graph():
	df = pd.read_csv("../data/metadata_clean.csv")
	n = len(df)
	nodes = {}

	for i in range(n):
		row_data = df.iloc[i]
		if type(row_data['sha']) == str and type(row_data['title']) == str:
			nodes[row_data['title']] = [row_data['abstract'], row_data['sha']]

	node_set = set()
	edges = read_edge_list_txt()
	for i, edge in enumerate(edges):
		head = edge[0]
		tail = edge[1]
		node_set.add(head)
		node_set.add(tail)
	print("number of nodes:", len(node_set))
	print("number of edges:", len(edges))

	G = nx.DiGraph()
	mapping = {}
	node_set = list(node_set)
	for i in range(len(node_set)):
		title = node_set[i]
		abstract = nodes[title][0]
		sha = nodes[title][1]
		mapping[title] = i
		G.add_node(i, title=title, abstract=abstract, sha=sha)

	for i, edge in enumerate(edges):
		head = edge[0]
		tail = edge[1]
		head_id = mapping[head]
		tail_id = mapping[tail]
		G.add_edge(head_id, tail_id, edge_type="reference")

	nx.write_gpickle(G, "../data/reference.gpickle")

def clean_gpickle():
	G = nx.read_gpickle("../data/reference.gpickle")
	H = G.copy()
	for node in G.nodes(data=True):
		if type(node[1]['abstract']) == float:
			print(node)
			H.remove_node(node[0])
	
	nx.write_gpickle(H, "../data/reference_clean.gpickle")
