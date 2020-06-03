import matplotlib
matplotlib.use("agg")
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

plt.rcParams['axes.titlesize'] = 10

# model = 'GAT'
model = 'SAGE'
path_template = '../figure/log/{}_{}_{}.pkl'

entity = None
no_entity = None
entity_min = None
entity_max = None
no_entity_min = None
no_entity_max = None

def plot(a, b, entity_min, entity_max, no_entity_min, no_entity_max):
	plt.figure()
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
	x = np.arange(1, 200, dtype=int)

	ax1.plot(x, a, label='LIKE')
	ax1.plot(x, b, label='without')
	ax1.set_xlabel('epochs')
	ax1.set_ylabel('accuracy')
	ax1.set_title('')
	ax1.grid(ls='--', color='black', lw=0.5, alpha=0.5)
	ax1.set_title('Average Accuracy')
	ax1.legend()
	
	ax2.plot(x, entity_min, color='#1f77b4')
	ax2.plot(x, entity_max, color='#1f77b4')
	ax2.fill_between(x, entity_max, entity_min, alpha=0.2, label='LIKE')
	ax2.plot(x, no_entity_min, color='#ff7f0e')
	ax2.plot(x, no_entity_max, color='#ff7f0e')
	ax2.fill_between(x, no_entity_max, no_entity_min, alpha=0.2, label='without')
	ax2.set_xlabel('epochs')
	ax2.set_ylabel('accuracy')
	ax2.grid(ls='--', color='black', lw=0.5, alpha=0.5)
	ax2.set_title('Min and Max Accuracy')
	ax2.legend()

	plt.tight_layout()
	plt.gcf().subplots_adjust(bottom=0.15)

	plt.savefig("../figure/{}.pdf".format(model))

for i in range(5):
	filename = path_template.format(model, i, True)
	with open(filename, 'rb') as f:
		data = pkl.load(f)
		curr_entity = np.array(data['test_accu'])
		if i == 0:
			entity = curr_entity.copy()
			entity_min = curr_entity.copy()
			entity_max = curr_entity.copy()
		else:
			entity += curr_entity
			entity_min = np.minimum(curr_entity, entity_min)
			entity_max = np.maximum(curr_entity, entity_max)
entity /= 5

for i in range(5):
	filename = path_template.format(model, i, False)
	with open(filename, 'rb') as f:
		data = pkl.load(f)
		curr_entity = np.array(data['test_accu'])
		if i == 0:
			no_entity = curr_entity.copy()
			no_entity_min = curr_entity.copy()
			no_entity_max = curr_entity.copy()
		else:
			no_entity += curr_entity
			no_entity_min = np.minimum(curr_entity, no_entity_min)
			no_entity_max = np.maximum(curr_entity, no_entity_max)
no_entity /= 5

plot(entity, no_entity, entity_min, entity_max, no_entity_min, no_entity_max)
