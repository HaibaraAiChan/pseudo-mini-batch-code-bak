import argparse
import dgl
import dgl.function as fn
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from dgl.utils import expand_as_pair
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import tqdm
import sys
sys.path.insert(0,'..')
from utils import Logger

class GraphSAGE(nn.Module):
	def __init__(self,
				 in_feats,
				 hidden_feats,
				 out_feats,
				 aggre,
				 num_layers,
				 activation,
				 dropout):
		super(GraphSAGE, self).__init__()
		self.n_hidden = hidden_feats
		self.n_classes = out_feats
		self.layers = nn.ModuleList()
		# self.bns = nn.ModuleList()
		# input layer
		self.layers.append(SAGEConv(in_feats, hidden_feats, aggre))
		# hidden layers
		for _ in range(num_layers - 2):
			self.layers.append(SAGEConv(hidden_feats, hidden_feats, aggre))
		# output layer
		self.layers.append(SAGEConv(hidden_feats, out_feats, aggre))
		self.dropout = nn.Dropout(p=dropout)
		self.activation = activation

	def reset_parameters(self):
		for layer in self.layers:
			layer.reset_parameters()

	def forward(self, blocks, x):
		for i, (layer, block) in enumerate(zip(self.layers[:-1], blocks[:-1])):
		# for i, layer in enumerate(self.layers[:-1]):
			x = layer(block, x)
			x = self.activation(x)
			x = self.dropout(x)
		x = self.layers[-1](blocks[-1], x)

		return x.log_softmax(dim=-1)
	
	def inference(self, g, x, args, device):
		"""
		Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
		g : the entire graph.
		x : the input of entire node set.

		The inference code is written in a fashion that it could handle any number of nodes and
		layers.
		"""
		# During inference with sampling, multi-layer blocks are very inefficient because
		# lots of computations in the first few layers are repeated.
		# Therefore, we compute the representation of all nodes layer by layer.  The nodes
		# on each layer are of course splitted in batches.
		# TODO: can we standardize this?
		# device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
		for l, layer in enumerate(self.layers):
			y = torch.zeros(g.num_nodes(), self.n_hidden if l!=len(self.layers) - 1 else self.n_classes)

			sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
			dataloader = dgl.dataloading.NodeDataLoader(
				g,
				torch.arange(g.num_nodes(),dtype=torch.long).to(device),
				sampler,
				# batch_size=24,
				batch_size=args.batch_size,
				shuffle=True,
				drop_last=False,
				num_workers=args.num_workers)


			for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
				block = blocks[0]
				block = block.int().to(device)
				h = x[input_nodes].to(device)
				h = layer(block, h)
				
				y[output_nodes] = h.cpu()

			x = y
		return y
