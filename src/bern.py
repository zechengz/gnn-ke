import time
import json
import requests
import networkx as nx

def query_raw(text, url="https://bern.korea.ac.kr/plain"):
	return requests.post(url, data={'sample_text': text}).json()

G = nx.read_gpickle("../data/reference_tensor.gpickle")
for i in range(G.number_of_nodes()):
	print(i)
	text = G.nodes[i]['abstract_raw']
	text = text.replace('Abstract', '')
	text = text.replace('abstract', '')
	text = text.strip()
	res = query_raw(text)
	if 'denotations' not in res:
		res = {'denotations': []}
	with open('../data/bern_entity/{}.json'.format(i), 'w') as f:
		json.dump(res['denotations'], f, indent=4)
	time.sleep(1)
