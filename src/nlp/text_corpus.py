import re
import nltk
import matplotlib
matplotlib.use("agg")
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from nltk.corpus import stopwords

PATH_MAIN = "../../"
threshold = 70

def text_process():
	G = nx.read_gpickle(PATH_MAIN + "data/reference.gpickle")
	text = ""
	n = G.number_of_nodes()
	for i, node in enumerate(G.nodes(data=True)):
		if i != n - 1 and i != 0:
			text += " "
		text += node[1]['title']
	sw = set(stopwords.words('english'))
	
	text = text.lower()
	text = text.strip()
	text = text.replace("\n", "")
	text = text.replace("\r", "")
	text = text.replace("\t", "")
	text = re.sub(r'[^\w\s]','',text)
	text = text.strip()
	text = text.split(" ")
	frequency = {}
	for word in text:
		if word in sw:
			continue
		if word.isdigit():
			continue
		if len(word) <= 2:
			continue
		if word in frequency:
			frequency[word] += 1
		else:
			frequency[word] = 1
	text_result = set(filter(lambda x: frequency[x] >= threshold, frequency))
	
	words = ""
	for word in text:
		if word in text_result:
			words += word
			words += " "

	mask = np.array(Image.open(PATH_MAIN + "figure/virus.jpg"))
	image_colors = ImageColorGenerator(mask)
	wordcloud = WordCloud(stopwords=set(), height=1440, width=1440, scale=5, mode="RGBA", 
							background_color="white", max_words=5000, mask=mask).generate(words)
	wordcloud.recolor(color_func=image_colors)
	wordcloud.to_file(PATH_MAIN + "figure/wordcloud.png")
	plt.figure(figsize=[12,12])
	plt.imshow(wordcloud, interpolation='bilinear', cmap=plt.cm.gray)
	plt.axis("off")

def save_text_txt():
	G = nx.read_gpickle(PATH_MAIN + "data/reference.gpickle")
	titles = []
	abstracts = []
	n = G.number_of_nodes()
	for i, node in enumerate(G.nodes(data=True)):
		titles.append(str(node[1]['title']))
		abstracts.append(str(node[1]['abstract']))

	with open(PATH_MAIN + "data/title.txt", "w") as f:
		for title in titles:
			f.write(title)
			f.write("\n")

	with open(PATH_MAIN + "data/abstract.txt", "w") as f:
		for abstract in abstracts:
			f.write(abstract)
			f.write("\n")

if __name__ == "__main__":
	# text_process()
	save_text_txt()
