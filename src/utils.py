import re
import json
import numpy as np
import networkx as nx
import pandas as pd
import torch
import pickle as pkl

from os import listdir
from os.path import isfile, join

from nltk.corpus import stopwords

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

def transform_nx():
	G = nx.read_gpickle("../data/reference_clean.gpickle")
	keys = list(G.nodes)
	vals = list(range(G.number_of_nodes()))
	mapping = dict(zip(keys, vals))
	G = nx.relabel_nodes(G, mapping)
	H = G.copy()
	for node in G.nodes(data=True):
		text = node[1]['abstract']
		text = text.lower()
		text = text.strip()
		text = text.replace("\n", "")
		text = text.replace("\r", "")
		text = text.replace("\t", "")
		text = re.sub(r'[^\w\s]','',text)
		text = text.strip()
		text = text.split(" ")
		if 'abstract' in text[0] or 'background' in text[0]:
			text = text[1:len(text)]
		if len(text) <= 2:
			H.remove_node(node[0])

	keys = list(H.nodes)
	vals = list(range(H.number_of_nodes()))
	mapping = dict(zip(keys, vals))
	H = nx.relabel_nodes(H, mapping)

	for node in H.nodes(data=True):
		text = node[1]['abstract']
		text = text.lower()
		text = text.strip()
		text = text.replace("\n", "")
		text = text.replace("\r", "")
		text = text.replace("\t", "")
		text = re.sub(r'[^\w\s]','',text)
		text = text.strip()
		text = text.split(" ")
		if 'abstract' in text[0] or 'background' in text[0]:
			text = text[1:len(text)]
		node[1]['abstract_raw'] =  node[1]['abstract']
		node[1]['abstract'] = text

	unique_word_dict = {}
	threshold = 200
	sw = set(stopwords.words('english'))
	for node in H.nodes(data=True):
		abstract = node[1]['abstract']
		for word in abstract:
			if word not in sw:
				if word in unique_word_dict:
					unique_word_dict[word] += 1
				else:
					unique_word_dict[word] = 1
	unique_word_list = [k for k, v in unique_word_dict.items() if v >= threshold]
	sorted_unique_word = sorted(unique_word_list)
	mapping = list(zip(range(len(sorted_unique_word)), sorted_unique_word))
	mapping = dict((v, k) for k, v in mapping)
	
	res = H.__class__()
	res.add_nodes_from(H)
	res.add_edges_from(H.edges)
	feature_dict = {}
	abstract_raw_dict = {}

	feature_dim = len(unique_word_list)
	print("Number of unique words:", feature_dim)
	count_zero = 0
	for node in H.nodes(data=True):
		feature = torch.zeros([feature_dim, ], dtype=torch.float32)
		abstract = node[1]['abstract']
		for word in abstract:
			if word not in sw and word in mapping:
				pos = mapping[word]
				feature[pos] = 1
		node[1]['node_feature'] = feature
		feature_dict[node[0]] = {'node_feature': feature}
		abstract_raw_dict[node[0]] = {'abstract_raw': node[1]['abstract_raw']}
		if feature.sum() == 0:
			count_zero += 1
		# print(feature.sum())
	nx.set_node_attributes(res, feature_dict)
	nx.set_node_attributes(res, abstract_raw_dict)

	
	features = nx.get_node_attributes(res,'node_feature')
	features_check = nx.get_node_attributes(H,'node_feature')
	assert len(features) == len(features_check)
	for node in features:
		# print(node, features[node])
		assert torch.all(features[node] == features_check[node]).item() == True

	print("Number of nodes:", H.number_of_nodes())
	print("Number of edges:", H.number_of_edges())
	print("Number of zero features:", count_zero)
	assert H.number_of_nodes() == res.number_of_nodes()
	assert H.number_of_edges() == res.number_of_edges()

	for node in res.nodes(data=True):
		print(node)
	nx.write_gpickle(res, "../data/reference_tensor.gpickle")

def list_file_names(repo):
	res = {}
	for i, file in enumerate(listdir(repo)):
		file_id = int(file[:-5])
		if file_id >= 10533:
			continue
		if '.json' in file:
			res[file_id] = file
	return res

def generate_pickle_from_json():
	repo_path = '../data/bern_entity/'
	id_dict = list_file_names(repo_path)
	entities = {}
	for file_id in id_dict:
		fname = repo_path + str(file_id) + '.json'
		with open(fname, 'r') as f:
			data = json.load(f)
			entities[file_id] = set()
			for item in data:
				entity = tuple(sorted(item['id']))
				entities[file_id].add(entity)
	with open('../data/bern.pkl', 'wb') as f:
		pkl.dump(entities, f)