import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy
import torch
import dgl
from pyvis.network import Network

matplotlib.use('Agg')


def merge_(list1, list2):
	merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
	return merged_list

def Diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))


def gen_pyvis_graph_local(block_to_graph, epoch, layer):
	G=block_to_graph

	induced_src = G.srcdata['_ID']
	# print('src nodes')
	
	# print(sorted(induced_src.tolist()))
	# print(induced_src)
	induced_dst = G.dstdata['_ID']
	src_nid_list = G.srcdata['_ID'].tolist()


	# dict_nid_2_global = {i: src_nid_list[i] for i in range(0, len(src_nid_list))}
	# eids_global_list = G.edata['_ID'].tolist()
	# print(eids_global_list)

	
	src_local, dst_local = G.edges(order='eid')

	src_g, dst_g = induced_src[src_local], induced_src[dst_local]

	sources = src_g.detach().numpy()
	targets = dst_g.detach().numpy()

	combination = [p for p in zip(sources, targets)]
	# print(combination)

	edge_data = zip(sources, targets)
	
	pyvis_net = Network(height='750px', width='100%', directed=True)
	for u in induced_src.tolist():
		if u not in induced_dst.tolist():
			pyvis_net.add_node(u, label=str(u))
	for v in induced_dst.tolist():
		pyvis_net.add_node(v, label=str(v),color='#dd4b39' )

	for e in edge_data:
		
		src = int(e[0])
		dst = int(e[1])

		# pyvis_net.add_node(1, label="Node 1")

		# pyvis_net.add_node(src)
		# pyvis_net.add_node(dst)
		if dst in induced_dst:
			# pyvis_net.add_node(dst, color='#dd4b39')
			pyvis_net.add_edge(src, dst, color='#dd4b39')
		else:
			# pyvis_net.add_node(dst)
			pyvis_net.add_edge(src, dst)

	pyvis_net.barnes_hut()
	pyvis_net.show_buttons(filter_=['physics'])

	file_name='figures/full_batch_subgraph/Epoch_'+str(epoch)+'_block_'+str(layer)+'_nx.html'
	pyvis_net.show(file_name)
	return 

def draw_dataloader_blocks_pyvis(blocks, epoch, batch_id):
	print('*-'*40)
	print(blocks)
	color_=['#008000','#069AF3','#dd4b39','#162347'] # blue, red, black
	# blocks.reverse()
	
	l=len(blocks)
	for b_id in range(len(blocks)):
		block=blocks[b_id]
		induced_src = block.srcdata[dgl.NID]
		induced_dst = block.dstdata[dgl.NID]
		pure_src= list(set(induced_src.tolist()) - set(induced_dst.tolist()))

		induced_eid = block.edata[dgl.NID].tolist()

		print('global src and dst nids')
		print(induced_src)
		print(induced_dst)
		e_src_local, e_dst_local = block.edges(order='eid')
		e_src, e_dst = induced_src[e_src_local], induced_src[e_dst_local]
		# e_src = e_src.detach().numpy().astype(int)
		# e_dst = e_dst.detach().numpy().astype(int)

		nt = Network(height='750px', width='100%', directed=True)
		for  u in pure_src:
			nt.add_node(int(u), label=str(u),size=20,color=color_[b_id])
		for  v in induced_dst.tolist():
			nt.add_node(int(v), label=str(v),size=20,color=color_[b_id+1])

		for eid, (u, v) in enumerate(zip(e_src, e_dst)):
			nt.add_edge(int(u), int(v), id=int(induced_eid[eid]), color=color_[b_id+1])
			
		
		nt.barnes_hut()
		nt.show_buttons(filter_=['physics'])
		file_name='figures/blocks/epoch_'+str(epoch)+'_batch_'+str(batch_id)+'_layer_'+str(b_id)+'_block_nx.html'
		nt.show(file_name)
	# blocks.reverse()
	return



def draw_dataloader_blocks_pyvis_total(blocks, batch_id):
	print('*-'*40)
	# print(blocks)
	color_=['#069AF3','#dd4b39','#162347'] # blue, red, black
	nt = Network(height='750px', width='100%', directed=True)
	# blocks.reverse()
	pre_edges_num=0
	blocks=block_[1]+block_[0]
	l=len(blocks)
	for layer, block in enumerate(blocks):
		induced_src = block.srcdata[dgl.NID]
		induced_dst = block.dstdata[dgl.NID]
		print('src and dst nids')
		print(induced_src)
		print(induced_dst)
		src_local, dst_local = block.edges(order='eid')
		src, dst = induced_src[src_local], induced_src[dst_local]
		src = src.detach().numpy().astype(int)
		dst = dst.detach().numpy().astype(int)
		dif=pre_edges_num
		for eid, (u, v) in enumerate(zip(src, dst)):
			# print(u)
			# print(type(u))
			# nx_graph.add_node(1, label="Node 1")
			if not nt.get_node(int(u)):
				nt.add_node(int(u), label=str(u),size=20,color=color_[l-layer])
			
			if not nt.get_node(int(v)):
				nt.add_node(int(v), label=str(v),size=20,color=color_[l-layer-1])
			nt.add_edge(int(u), int(v), id=int(eid+dif),color=color_[l-layer-1])
			pre_edges_num=eid
	nt.barnes_hut()
	nt.show_buttons(filter_=['physics'])
	file_name='figures/blocks/batch_'+str(batch_id)+'_all_layer_blocks_nx.html'
	nt.show(file_name)
	# blocks.reverse()
	return


def draw_dataloader_block_nx_pyvis(blocks, batch_id):
	print('*-'*40)
	print(blocks)
	
	for layer, block in enumerate(blocks):
		induced_src = block.srcdata[dgl.NID]
		induced_dst = block.dstdata[dgl.NID]
		print('src and dst nids')
		print(induced_src)
		print(induced_dst)
		src_local, dst_local = block.edges(order='eid')
		src, dst = induced_src[src_local], induced_src[dst_local]
		src = src.detach().numpy().astype(int)
		dst = dst.detach().numpy().astype(int)
		#------------------------------------------------------------------dgl block to networkx graph
		nx_graph = nx.MultiDiGraph()
		# nx_graph.add_nodes_from(range(g.number_of_nodes()))
		for eid, (u, v) in enumerate(zip(src, dst)):
			# print(u)
			# print(type(u))
			# nx_graph.add_node(1, label="Node 1")
			nx_graph.add_node(int(u), label=str(u),size=20,labelHighlightBold=True)
			nx_graph.add_node(int(v), label=str(v),size=20,labelHighlightBold=True)
			nx_graph.add_edge(int(u), int(v), id=int(eid), arrowStrikethrough=True)
		#------------------------------------------------------------------------------pyvis
		nt = Network(height='750px', width='100%', directed=True)
		nt.barnes_hut()
		nt.show_buttons(filter_=['physics'])

		nt.from_nx(nx_graph)
		file_name='figures/blocks/batch_'+str(batch_id)+'_layer_'+str(layer)+'_block_nx.html'
		nt.show(file_name)


# def gen_pyvis_graph_global(global_G, train_nid):
# 	G=global_G
# 	print(G)	
	
# 	sources_, targets_ = G.edges()
	
# 	sources=sources_.detach().numpy()
# 	targets=targets_.detach().numpy()
# 	edge_data = zip(sources, targets)
# 	combination = [p for p in zip(sources, targets)]
# 	print('raw graph edges: ')
# 	print(combination)

# 	pyvis_net = Network(height='750px', width='100%', directed=True)
# 	for u in range(24,34):
# 		pyvis_net.add_node(u, label=str(u))
# 	for v in range(24):
# 		pyvis_net.add_node(v, label=str(v),color='#dd4b39' )

# 	for e in edge_data:
# 		src = int(e[0])
# 		dst = int(e[1])
# 		# pyvis_net.add_node(1, label="Node 1")
# 		# pyvis_net.add_node(src, label=str(src))
# 		# pyvis_net.add_node(dst, label=str(dst))

# 		if dst in train_nid.tolist():
# 			pyvis_net.add_edge(src, dst, color='#dd4b39')
# 		else:
# 			pyvis_net.add_edge(src, dst)
	
# 	pyvis_net.barnes_hut()
# 	pyvis_net.show_buttons(filter_=['physics'])

# 	file_name='figures/raw_graph_nx.html'
# 	pyvis_net.show(file_name)
# 	return 

def gen_pyvis_graph_global(global_G, train_nid):
	G=global_G
	print(G)	
	
	sources_, targets_ = G.edges()
	
	sources=sources_.detach().numpy()
	targets=targets_.detach().numpy()
	edge_data = zip(sources, targets)
	combination = [p for p in zip(sources, targets)]
	print('raw graph edges: ')
	print(combination)

	pyvis_net = Network(height='750px', width='100%', directed=True)
	for u in range(4,34):
		pyvis_net.add_node(u, label=str(u))
	for v in range(4):
		pyvis_net.add_node(v, label=str(v),color='#dd4b39' )

	for e in edge_data:
		src = int(e[0])
		dst = int(e[1])
		# pyvis_net.add_node(1, label="Node 1")
		# pyvis_net.add_node(src, label=str(src))
		# pyvis_net.add_node(dst, label=str(dst))

		if dst in train_nid.tolist():
			pyvis_net.add_edge(src, dst, color='#dd4b39')
		else:
			pyvis_net.add_edge(src, dst)
	
	pyvis_net.barnes_hut()
	pyvis_net.show_buttons(filter_=['physics'])

	file_name='figures/raw_graph_nx.html'
	pyvis_net.show(file_name)
	return 




def draw_graph(G, epoch):
	
	fig = plt.figure()
	black_edges = G.edges()
	black_edges = list(black_edges)
	print(black_edges[0])
	print(black_edges[1])
	# print('total eid number   '	)
	# print(len(black_edges[0]))
	# dd = int(len(black_edges[0]) / 2)
	dd = int(len(black_edges[0]))
	black_edges[0] = black_edges[0].tolist()
	black_edges[1] = black_edges[1].tolist()
	black_edges = merge_(black_edges[0][:dd], black_edges[1][:dd])
	# print('black_edges')
	# print(black_edges)
	nx_G = G.to_networkx()
	# nx_G = G.to_networkx().to_undirected()

	# pos = nx.kamada_kawai_layout(nx_G)
	pos = nx.spring_layout(nx_G)

	nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
	# nx.draw_networkx_edge_labels(nx_G, pos, font_color='r', label_pos=0.7)
	# nx.draw_networkx_edges(nx_G, pos,  arrows=False)
	nx.draw_networkx_edges(nx_G, pos, edgelist=black_edges, arrows=True)
	ax = plt.gca()
	ax.margins(0.20)

	plt.axis("off")
	# plt.show()
	plt.savefig('TTTTTTTTT karate full batch sub-graph.eps',format='eps')
	return

	

def draw_nx(nx_G, epoch):
	fig = plt.figure()
	pos = nx.spring_layout(nx_G)

	nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
	# nx.draw_networkx_edge_labels(nx_G, pos, font_color='r', label_pos=0.7)
	# nx.draw_networkx_edges(nx_G, pos,  arrows=False)
	# nx.draw_networkx_edges(nx_G,  edgelist=black_edges, arrows=True)
	ax = plt.gca()
	ax.margins(0.20)

	plt.axis("off")
	plt.show()
	plt.savefig('figures/'+str(epoch)+'_TTTTTTTTT karate full batch sub-graph_networkx.jpg',format='jpg')
	return
	
	
def dgl_to_networkx(g,epoch):
	if not g.is_homogeneous:
		raise DGLError('dgl.to_networkx only supports homogeneous graphs.')
	print(g)
	#homogeneous graph has no dstdata
	induced_src = g.ndata['_ID']
	# TODO
	# it has errors
	print('src nids')
	print(induced_src)
		
	src_local, dst_local = g.edges(order='eid')
	src, dst = induced_src[src_local], induced_dst[dst_local]

	print()
	src = src.detach().numpy().astype(int)
	dst = dst.detach().numpy().astype(int)
	# xiangsx: Always treat graph as multigraph
	nx_graph = nx.MultiDiGraph()
	nx_graph.add_nodes_from(range(g.number_of_nodes()))
	for eid, (u, v) in enumerate(zip(src, dst)):
		# print(eid)
		nx_graph.add_node(u, label=str(u),size=20,labelHighlightBold=True)
		nx_graph.add_node(v, label=str(v),size=20,labelHighlightBold=True)
		nx_graph.add_edge(int(u), int(v), id=int(eid), arrowStrikethrough=True)
	# nx.draw(nx_graph)
	draw_nx(nx_graph,epoch)
	
	return nx_graph


def generate_interactive_graph(G, epoch):
	
	
	nt = Network(height='750px', width='100%', directed=True)
	nx_graph=dgl_to_networkx(G, epoch)
	# nx_graph = G.to_networkx()
	# nx_graph = nx.complete_graph(5)
	nt.barnes_hut()
	nt.show_buttons(filter_=['physics'])

	nt.from_nx(nx_graph)
	file_name='figures/full_batch_subgraph/'+str(epoch)+'_nx.html'
	nt.show(file_name)
	
