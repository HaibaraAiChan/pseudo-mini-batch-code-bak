import torch
import dgl
import numpy
import time
import pickle
import io
from math import ceil
from math import floor
from itertools import islice
from statistics import mean
from multiprocessing import Manager, Pool
from multiprocessing import Process, Value, Array
# from graph_partitioner import Graph_Partitioner
from my_utils import gen_batch_output_list
from draw_graph import draw_graph, draw_dataloader_blocks_pyvis,draw_dataloader_blocks_pyvis_total
from memory_usage import see_memory_usage



# from draw_nx import draw_nx_graph
from sortedcontainers import SortedList, SortedSet, SortedDict
from multiprocessing import Process, Queue
from collections import Counter, OrderedDict

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)
#------------------------------------------------------------------------
def unique_tensor_item(combined):
	uniques, counts = combined.unique(return_counts=True)
	return uniques.type(torch.long)


def unique_edges(edges_list):
	temp = []
	for i in range(len(edges_list)):
		tt = edges_list[i]  # tt : [[],[]]
		for j in range(len(tt[0])):
			cur = (tt[0][j], tt[1][j])
			if cur not in temp:
				temp.append(cur)
	# print(temp)   # [(),(),()...]
	res_ = list(map(list, zip(*temp)))  # [],[]
	res = tuple(sub for sub in res_)
	return res


def generate_random_mini_batch_seeds_list(OUTPUT_NID, args):
	'''
	Parameters
	----------
	OUTPUT_NID: final layer output nodes id (tensor)
	args : all given parameters collection

	Returns
	-------
	'''
	selection_method = args.selection_method
	mini_batch = args.batch_size
	full_len = len(OUTPUT_NID)  # get the total number of output nodes
	if selection_method == 'random':
		indices = torch.randperm(full_len)  # get a permutation of the index of output nid tensor (permutation of 0~n-1)
	else: #selection_method == 'range'
		indices = torch.tensor(range(full_len))

	output_num = len(OUTPUT_NID.tolist())
	map_output_list = list(numpy.array(OUTPUT_NID)[indices.tolist()])
	batches_nid_list = [map_output_list[i:i + mini_batch] for i in range(0, len(map_output_list), mini_batch)]
	weights_list = []
	for i in batches_nid_list:
		temp = len(i)/output_num
		weights_list.append(len(i)/output_num)
		
	return batches_nid_list, weights_list

def get_global_graph_edges_ids_block(raw_graph, block):
	
	edges=block.edges(order='eid', form='all')
	edge_src_local = edges[0]
	edge_dst_local = edges[1]
	# edge_eid_local = edges[2]
	induced_src = block.srcdata[dgl.NID]
	induced_dst = block.dstdata[dgl.NID]
	induced_eid = block.edata[dgl.EID] 
		
	raw_src, raw_dst=induced_src[edge_src_local], induced_dst[edge_dst_local]
	# raw_src, raw_dst=induced_src[edge_src_local], induced_src[edge_dst_local]
	
	# in homo graph: raw_graph 
	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
	# https://docs.dgl.ai/generated/dgl.DGLGraph.edge_ids.html?highlight=graph%20edge_ids#dgl.DGLGraph.edge_ids
	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

	return global_graph_eids_raw, (raw_src, raw_dst)

def get_global_graph_edges_ids_2(raw_graph, block_to_graph):
	
	edges=block_to_graph.edges(order='eid')
	edge_src_local = edges[0]
	edge_dst_local = edges[1]
	induced_src = block_to_graph.srcdata[dgl.NID]
	induced_dst = block_to_graph.dstdata[dgl.NID]
		
	raw_src, raw_dst=induced_src[edge_src_local], induced_dst[edge_dst_local]
	# raw_src = block_to_graph.ndata[dgl.NID]['_N_src'][src] 
	# raw_dst= block_to_graph.ndata[dgl.NID]['_N_dst'][dst]
	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

	return global_graph_eids_raw, (raw_src, raw_dst)



def get_global_graph_edges_ids(raw_graph, cur_block):
	'''
		Parameters
		----------
		raw_graph : graph
		cur_block: (local nids, local nids): (tensor,tensor)

		Returns
		-------
		global_graph_edges_ids: []                    current block edges global id list
	'''
	src, dst = cur_block.all_edges(order='eid')
	src = src.long()
	dst = dst.long()
	raw_src, raw_dst = cur_block.srcdata[dgl.NID][src], cur_block.dstdata[dgl.NID][dst]
	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids
	return global_graph_eids_raw, (raw_src, raw_dst)


def generate_one_block(raw_graph, global_eids, global_srcnid, global_dstnid):
	'''

	Parameters
	----------
	G    global graph                     DGLGraph
	eids  cur_batch_subgraph_global eid   tensor int64

	Returns
	-------

	'''
	_graph = dgl.edge_subgraph(raw_graph, global_eids,store_ids=True)
	edge_src_list = _graph.edges(order='eid')[0].tolist()
	edge_dst_list = _graph.edges(order='eid')[1].tolist()
	eid_list = _graph.edata['_ID'].tolist()

	dst_local_nid_list=list(Counter(edge_dst_list).keys())
	# dst_local_nid_list=[]
	# [dst_local_nid_list.append(nid) for nid in edge_dst_list if nid not in dst_local_nid_list]
	# to keep the order of dst nodes
	new_block = dgl.to_block(_graph, dst_nodes=torch.tensor(dst_local_nid_list, dtype=torch.long))
	if set(_graph.ndata[dgl.NID].tolist())!=set(global_srcnid.tolist()):
		print()
	if len(new_block.srcdata[dgl.NID])!=len(global_srcnid.tolist()):
		print('not match')
		print()
		return 
	new_block.srcdata[dgl.NID] = global_srcnid
	new_block.dstdata[dgl.NID] = global_dstnid
	# print(new_block.edata['_ID'].tolist())
	new_block.edata['_ID']=_graph.edata['_ID']


	return new_block

def check_connections_0(batched_nodes_list, current_layer_subgraph):
	res=[]
	induced_src = current_layer_subgraph.srcdata[dgl.NID]
	induced_dst = current_layer_subgraph.dstdata[dgl.NID]
	eids_global = current_layer_subgraph.edata['_ID']
	# print('current layer subgraph eid (global)')
	# print(sorted(eids_global.tolist()))

	src_nid_list = induced_src.tolist()
	src_local, dst_local, index = current_layer_subgraph.edges(form='all')
	# print( current_layer_subgraph.edges(form='all')[0])
	# print( current_layer_subgraph.edges(form='all')[1])
	# print( current_layer_subgraph.edges(form='all')[2])
	# src_local, dst_local = current_layer_subgraph.edges(order='eid')

	src, dst = induced_src[src_local], induced_src[dst_local]
	dict_nid_2_local = {src_nid_list[i]: i for i in range(0, len(src_nid_list))}
	
	src_compare=[]
	dst_compare=[]
	compare=[]
	prev_eids=[]
	for step, output_nid in enumerate(batched_nodes_list):
		# in current layer subgraph, only has src and dst nodes,
		# and src nodes includes dst nodes, src nodes equals dst nodes.
		local_output_nid = list(map(dict_nid_2_local.get, output_nid))
		
		local_in_edges_tensor = current_layer_subgraph.in_edges(local_output_nid, form='all')
		# return (𝑈,𝑉,𝐸𝐼𝐷)
		# get local srcnid and dstnid from subgraph
		mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
		
		mini_batch_dst_local= list(local_in_edges_tensor)[1]
		mini_batch_dst_global= induced_src[mini_batch_dst_local].tolist()
		if set(mini_batch_dst_local.tolist()) != set(local_output_nid):
			print('local dst not match')
		eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
		
		add_src=[i for i in mini_batch_src_global if i not in output_nid]
		r_ = []
		[r_.append(x) for x in add_src if x not in r_]
		src_nid = torch.tensor(output_nid + r_, dtype=torch.long)
		output_nid = torch.tensor(output_nid, dtype=torch.long)

		
		
		res.append((src_nid, output_nid, global_eid_tensor, local_output_nid))
		# res.append((src_nid, output_nid, global_eid_tensor, mini_batch_src_global, mini_batch_dst_global))
		compare.append(global_eid_tensor.tolist())
		src_compare.append(src_nid.tolist())
		dst_compare.append(output_nid.tolist())

	tttt=sum(compare,[])
	# print(sorted(list(set(tttt))))
	if set(tttt)!= set(eids_global.tolist()):
		print('the edges not match')
		print(sorted(list(set(tttt))))
		print(sorted(list(set(eids_global.tolist()))))
	if set(sum(src_compare,[]))!= set(induced_src.tolist()):
		print('the src nodes not match')
		print(set(sum(src_compare,[])))
		print(set(induced_src.tolist()))
	if set(sum(dst_compare,[]))!= set(induced_dst.tolist()):
		print('the dst nodes not match')
		print(set(sum(dst_compare,[])))
		print(set(induced_dst.tolist()))
	return res

	

def check_connections_block(batched_nodes_list, current_layer_block):
	res=[]
	induced_src = current_layer_block.srcdata[dgl.NID]
	induced_dst = current_layer_block.dstdata[dgl.NID]
	eids_global = current_layer_block.edata['_ID']
	# print('current layer subgraph eid (global)')
	# print(sorted(eids_global.tolist()))
	# return
	src_nid_list = induced_src.tolist()
	src_local, dst_local, index = current_layer_block.edges(form='all')


	src, dst = induced_src[src_local], induced_src[dst_local]
	dict_nid_2_local = {src_nid_list[i]: i for i in range(0, len(src_nid_list))}
	
	src_compare=[]
	dst_compare=[]
	compare=[]
	prev_eids=[]
	for step, output_nid in enumerate(batched_nodes_list):
		# in current layer subgraph, only has src and dst nodes,
		# and src nodes includes dst nodes, src nodes equals dst nodes.
		local_output_nid = list(map(dict_nid_2_local.get, output_nid))
		
		local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
		# return (𝑈,𝑉,𝐸𝐼𝐷)
		# get local srcnid and dstnid from subgraph
		mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
		
		mini_batch_dst_local= list(local_in_edges_tensor)[1]
		mini_batch_dst_global= induced_src[mini_batch_dst_local].tolist()
		if set(mini_batch_dst_local.tolist()) != set(local_output_nid):
			print('local dst not match')
		eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
		# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  bottleneck  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		
		c=OrderedCounter(mini_batch_src_global)
		list(map(c.__delitem__, filter(c.__contains__,output_nid)))
		
		r_=list(c.keys())

		# add_src=[i for i in mini_batch_src_global if i not in output_nid] 
		# r_ =remove_duplicate(add_src)
		# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$   bottleneck  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		#
		# [r_.append(x) for x in add_src if x not in r_]
		src_nid = torch.tensor(output_nid + r_, dtype=torch.long)
		output_nid = torch.tensor(output_nid, dtype=torch.long)

		
		
		res.append((src_nid, output_nid, global_eid_tensor, local_output_nid))
		# res.append((src_nid, output_nid, global_eid_tensor, mini_batch_src_global, mini_batch_dst_global))
		compare.append(global_eid_tensor.tolist())
		src_compare.append(src_nid.tolist())
		dst_compare.append(output_nid.tolist())

	tttt=sum(compare,[])
	# print(sorted(list(set(tttt))))
	# if set(tttt)!= set(eids_global.tolist()):
	# 	print('the edges not match')
	# 	print(sorted(list(set(tttt))))
	# 	print(sorted(list(set(eids_global.tolist()))))
	# if set(sum(src_compare,[]))!= set(induced_src.tolist()):
	# 	print('the src nodes not match')
	# 	print(set(sum(src_compare,[])))
	# 	print(set(induced_src.tolist()))
	# if set(sum(dst_compare,[]))!= set(induced_dst.tolist()):
	# 	print('the dst nodes not match')
	# 	print(set(sum(dst_compare,[])))
	# 	print(set(induced_dst.tolist()))
	if set(tttt)!= set(eids_global.tolist()):
		print('the edges not  match')
		return
		print(SortedSet(tttt))
		print(SortedSet(eids_global.tolist()))
		print(len(eids_global.tolist()))
	if set(sum(src_compare,[]))!= set(induced_src.tolist()):
		print('the src nodes  not  match')
		return
		print(SortedSet(sum(src_compare,[])))
		print(SortedSet(induced_src.tolist()))
		print(len(induced_src.tolist()))
	if set(sum(dst_compare,[]))!= set(induced_dst.tolist()):
		print('the dst nodes not match')
		return
		print(SortedSet(sum(dst_compare,[])))
		print(SortedSet(induced_dst.tolist()))
		print(len(induced_dst))
	return res



def generate_blocks_for_one_layer_2(raw_graph, block_2_graph, batches_nid_list):
	
	layer_src = block_2_graph.srcdata[dgl.NID]
	layer_dst = block_2_graph.dstdata[dgl.NID]
	layer_eid = block_2_graph.edata[dgl.NID].tolist()
	# print(sorted(layer_eid))
	
	blocks = []
	check_connection_time = []
	block_generation_time = []

	t1= time.time()
	batches_temp_res_list = check_connections_0(batches_nid_list, block_2_graph)
	t2 = time.time()
	check_connection_time.append(t2-t1) #------------------------------------------
	src_list=[]
	dst_list=[]
	ll=len(batches_temp_res_list)

	src_compare=[]
	dst_compare=[]
	eid_compare=[]
	for step, (srcnid, dstnid, current_block_global_eid) in enumerate(batches_temp_res_list):
	# for step, (srcnid, dstnid, current_block_global_eid, src_e, dst_e) in enumerate(batches_temp_res_list):
		# print('batch ' + str(step) + '-' * 30)
		t_ = time.time()
		if step == ll-1:
			print()
		# if len(prev_batched_eid_list) and prev_batched_eid_list[step]:
		# 	new_eids=current_block_global_eid.tolist()
		# 	pure_new_eid=[]
		# 	[pure_new_eid.append(eid) for eid in new_eids if eid not in pure_new_eid and eid not in prev_batched_eid_list[step]] #remove duplicate
		# 	current_block_global_eid=prev_batched_eid_list[step]+pure_new_eid
		# 	current_block_global_eid=torch.tensor(current_block_global_eid, dtype=torch.long)
		# 	cur_block = generate_one_block(raw_graph, current_block_global_eid, srcnid, dstnid)
		# else:
		# 	cur_block = generate_one_block(raw_graph, current_block_global_eid, srcnid, dstnid)
		cur_block = generate_one_block(raw_graph, current_block_global_eid, srcnid, dstnid)

		t__=time.time()
		block_generation_time.append(t__-t_)  #------------------------------------------
		#----------------------------------------------------
		# print('generate_blocks_for_one_layer function ------ batch: ', step)
		induced_src = cur_block.srcdata[dgl.NID]
		induced_dst = cur_block.dstdata[dgl.NID]
		induced_eid = cur_block.edata[dgl.NID].tolist()
		# print('src and dst nids')
		# print(induced_src)
		# print(induced_dst)
		e_src_local, e_dst_local = cur_block.edges(order='eid')
		e_src, e_dst = induced_src[e_src_local], induced_src[e_dst_local]
		e_src = e_src.detach().numpy().astype(int)
		e_dst = e_dst.detach().numpy().astype(int)

		combination = [p for p in zip(e_src, e_dst)]
		# print('batch block graph edges: ')
		# print(combination)
		#----------------------------------------------------
		blocks.append(cur_block)
		src_list.append(srcnid)
		dst_list.append(dstnid)

		eid_compare.append(induced_eid)
		src_compare.append(induced_src.tolist())
		dst_compare.append(induced_dst.tolist())

	tttt=sum(eid_compare,[])
	# print((set(tttt)))
	if set(tttt)!= set(layer_eid):
		print('the edges not match')
		print(sorted(list((set(tttt)))))
		print(sorted(list(set(layer_eid))))
	if set(sum(src_compare,[]))!= set(layer_src.tolist()):
		print('the src nodes not match')
		print(set(sum(src_compare,[])))
		print(set(layer_src.tolist()))
	if set(sum(dst_compare,[]))!= set(layer_dst.tolist()):
		print('the dst nodes not match')
		print(set(sum(dst_compare,[])))
		print(set(layer_dst.tolist()))

		# data_loader.append((srcnid, dstnid, [cur_block]))
		
	# print("\nconnection checking time " + str(sum(check_connection_time)))
	# print("total of block generation time " + str(sum(block_generation_time)))
	# print("average of block generation time " + str(mean(block_generation_time)))
	connection_time = sum(check_connection_time)
	block_gen_time = sum(block_generation_time)
	mean_block_gen_time = mean(block_generation_time)


	return blocks, src_list,dst_list,(connection_time, block_gen_time, mean_block_gen_time)


def generate_blocks_for_one_layer_block(raw_graph, layer_block, batches_nid_list):
	see_memory_usage("----------------------------------------before generate_blocks_for_one_layer_block ")
	layer_src = layer_block.srcdata[dgl.NID]
	layer_dst = layer_block.dstdata[dgl.NID]
	layer_eid = layer_block.edata[dgl.NID].tolist() # we have changed it to global eid
	# print(sorted(layer_eid))
	
	blocks = []
	check_connection_time = []
	block_generation_time = []

	t1= time.time()
	batches_temp_res_list = check_connections_block(batches_nid_list, layer_block)
	# return
	t2 = time.time()
	check_connection_time.append(t2-t1) #------------------------------------------
	src_list=[]
	dst_list=[]
	ll=len(batches_temp_res_list)

	src_compare=[]
	dst_compare=[]
	eid_compare=[]
	for step, (srcnid, dstnid, current_block_global_eid, local_dstnid) in enumerate(batches_temp_res_list):
	# for step, (srcnid, dstnid, current_block_global_eid, src_e, dst_e) in enumerate(batches_temp_res_list):
		# print('batch ' + str(step) + '-' * 30)
		t_ = time.time()

		cur_block = generate_one_block(raw_graph, current_block_global_eid, srcnid, dstnid)

		t__=time.time()
		block_generation_time.append(t__-t_)  #------------------------------------------
		#----------------------------------------------------
		# print('generate_blocks_for_one_layer function ------ batch: ', step)
		induced_src = cur_block.srcdata[dgl.NID]
		induced_dst = cur_block.dstdata[dgl.NID]
		induced_eid = cur_block.edata[dgl.NID].tolist()
	
		e_src_local, e_dst_local = cur_block.edges(order='eid')
		e_src, e_dst = induced_src[e_src_local], induced_src[e_dst_local]
		e_src = e_src.detach().numpy().astype(int)
		e_dst = e_dst.detach().numpy().astype(int)

		combination = [p for p in zip(e_src, e_dst)]
		# print('batch block graph edges: ')
		# print(combination)
		#----------------------------------------------------
		blocks.append(cur_block)
		src_list.append(srcnid)
		dst_list.append(dstnid)

		eid_compare.append(induced_eid)
		src_compare.append(induced_src.tolist())
		dst_compare.append(induced_dst.tolist())

	tttt=sum(eid_compare,[])
	# print((set(tttt)))
	if set(tttt)!= set(layer_eid):
		print('the edges not match !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		print(sorted(list((set(tttt)))))
		print(sorted(list(set(layer_eid))))
	if set(sum(src_compare,[]))!= set(layer_src.tolist()):
		print('the src nodes not match !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		print(set(sum(src_compare,[])))
		print(set(layer_src.tolist()))
	if set(sum(dst_compare,[]))!= set(layer_dst.tolist()):
		print('the dst nodes not match !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		print(set(sum(dst_compare,[])))
		print(set(layer_dst.tolist()))

		# data_loader.append((srcnid, dstnid, [cur_block]))
		
	
	connection_time = sum(check_connection_time)
	block_gen_time = sum(block_generation_time)
	# mean_block_gen_time = mean(block_generation_time)


	return blocks, src_list, dst_list, (connection_time, block_gen_time)





def eid_connect_check(current_layer_subgraph, B_src, B_dst, eids_to_compare):
	induced_src = current_layer_subgraph.srcdata[dgl.NID]
	induced_dst = current_layer_subgraph.dstdata[dgl.NID]
	eids_global = current_layer_subgraph.edata['_ID']
	src_local, dst_local, index = current_layer_subgraph.edges(form='all')
	src, dst = induced_src[src_local], induced_src[dst_local]
	src_nid_list = induced_src.tolist()
	dict_nid_2_local = {src_nid_list[i]: i for i in range(0, len(src_nid_list))}

	local_output_nid = list(map(dict_nid_2_local.get, B_dst.tolist()))
	
	local_in_edges_tensor = current_layer_subgraph.in_edges(local_output_nid, form='all')
		# return (𝑈,𝑉,𝐸𝐼𝐷)
		# get local srcnid and dstnid from subgraph
	mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
	mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
		
	mini_batch_dst_local= list(local_in_edges_tensor)[1]
	mini_batch_dst_global= induced_src[mini_batch_dst_local].tolist()
	if set(mini_batch_dst_local.tolist()) != set(local_output_nid):
		print('local dst not match')
	eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
	global_eid_tensor = eids_global[eid_local_list] # map local eid to global.

	add_src=[i for i in mini_batch_src_global if i not in B_dst.tolist()]
	r_ = []
	[r_.append(x) for x in add_src if x not in r_]
	src_nid = torch.tensor(B_dst.tolist() + r_, dtype=torch.long)
	
	src_____=[]
	[src_____.append(x) for x in mini_batch_src_global if x not in src_____]
	if src_____ != B_src.tolist():
		print('in function eid_connect_check: global src  not match ') 
	# if output_nid
	res=(global_eid_tensor == eids_to_compare)

		
	return res

def generate_blocks_for_one_layer(raw_graph, block_2_graph, batches_nid_list):

	layer_src = block_2_graph.srcdata[dgl.NID]
	layer_dst = block_2_graph.dstdata[dgl.NID]
	layer_eid = block_2_graph.edata[dgl.NID].tolist()
	# print(sorted(layer_eid))
	
	blocks = []
	check_connection_time = []
	block_generation_time = []

	t1= time.time()
	batches_temp_res_list = check_connections_0(batches_nid_list, block_2_graph)
	t2 = time.time()
	check_connection_time.append(t2-t1) #------------------------------------------
	src_list=[]
	dst_list=[]
	ll=len(batches_temp_res_list)

	src_compare=[]
	dst_compare=[]
	eid_compare=[]
	for step, (srcnid, dstnid, current_block_global_eid, local_dstnid) in enumerate(batches_temp_res_list):
	# for step, (srcnid, dstnid, current_block_global_eid, src_e, dst_e) in enumerate(batches_temp_res_list):
		# print('batch ' + str(step) + '-' * 30)
		t_ = time.time()
		if step == ll-1:
			print()
		#local index works in_subgraph
		# frontier=block_2_graph.in_subgraph({'_N_dst':local_dstnid})
		# print(frontier)
		# print(frontier.dstdata)
		# print(frontier.dstdata['_ID'])
		# print(frontier.srcdata)
		# print(frontier.srcdata['_ID'])
		# BB=dgl.to_block(frontier, dst_nodes={'_N_dst':local_dstnid})
		# print('block-------')
		# BB.dstdata['_ID']=BB.dstdata['_ID']['_N_dst']
		# BB.srcdata['_ID']=BB.srcdata['_ID']['_N_src']
		# print(BB)
		# print(BB.dstdata)
		# print(BB.srcdata)
		# print(BB.dstdata['_ID'])
		# print(BB.srcdata['_ID'])
		
		# print('layer_src[BB.srcdata[_ID]]')
		# print(BB.srcdata['_ID']['_N_dst'])
		# print(layer_src[BB.srcdata['_ID']['_N_src']])
		# print(layer_src[BB.srcdata['_ID']['_N_dst']])
		# to_check_src=layer_src[BB.srcdata['_ID']['_N_src']]
		# to_check_dst=layer_src[BB.srcdata['_ID']['_N_dst']]
		# # eids_to_compare=torch.tensor([16, 66, 14, 23], dtype=torch.long)
		# ttttt=eid_connect_check(block_2_graph, to_check_src, to_check_dst, current_block_global_eid)
		# print('eid connect check result')
		# print(ttttt)
		# print(BB.edata['_ID'])
		# print('-------------------------------------------')
		# dstnid=torch.tensor([0, 21], dtype=torch.long)
		# srcnid=torch.tensor([0, 21, 1, 11], dtype=torch.long)
		# current_block_global_eid=torch.tensor([16, 66, 14, 23], dtype=torch.long)
		cur_block = generate_one_block(raw_graph, current_block_global_eid, srcnid, dstnid)
		
		# print(cur_block)

		# cur_block = generate_one_block(raw_graph, current_block_global_eid, srcnid, dstnid)
		t__=time.time()
		block_generation_time.append(t__-t_)  #------------------------------------------
		#----------------------------------------------------
		# print('generate_blocks_for_one_layer function ------ batch: ', step)
		induced_src = cur_block.srcdata[dgl.NID]
		induced_dst = cur_block.dstdata[dgl.NID]
		induced_eid = cur_block.edata[dgl.NID].tolist()
		# print('src and dst nids')
		# print(induced_src)
		# print(induced_dst)
		e_src_local, e_dst_local = cur_block.edges(order='eid')
		e_src, e_dst = induced_src[e_src_local], induced_src[e_dst_local]
		e_src = e_src.detach().numpy().astype(int)
		e_dst = e_dst.detach().numpy().astype(int)

		combination = [p for p in zip(e_src, e_dst)]
		# print('batch block graph edges: ')
		# print(combination)
		#----------------------------------------------------
		blocks.append(cur_block)
		src_list.append(srcnid)
		dst_list.append(dstnid)

		eid_compare.append(induced_eid)
		src_compare.append(induced_src.tolist())
		dst_compare.append(induced_dst.tolist())

	tttt=sum(eid_compare,[])
	# print((set(tttt)))
	if set(tttt)!= set(layer_eid):
		print('the edges not match')
		print(sorted(list((set(tttt)))))
		print(sorted(list(set(layer_eid))))
	if set(sum(src_compare,[]))!= set(layer_src.tolist()):
		print('the src nodes not match')
		print(set(sum(src_compare,[])))
		print(set(layer_src.tolist()))
	if set(sum(dst_compare,[]))!= set(layer_dst.tolist()):
		print('the dst nodes not match')
		print(set(sum(dst_compare,[])))
		print(set(layer_dst.tolist()))

		# data_loader.append((srcnid, dstnid, [cur_block]))
		
	# print("\nconnection checking time " + str(sum(check_connection_time)))
	# print("total of block generation time " + str(sum(block_generation_time)))
	# print("average of block generation time " + str(mean(block_generation_time)))
	connection_time = sum(check_connection_time)
	block_gen_time = sum(block_generation_time)
	mean_block_gen_time = mean(block_generation_time)


	return blocks, src_list, dst_list, (connection_time, block_gen_time, mean_block_gen_time)

def generate_dataloader_w_partition(raw_graph, block_to_graph_list, args):
	for layer, block_to_graph in enumerate(block_to_graph_list):
		
		current_block_eidx_raw, current_block_edges_raw = get_global_graph_edges_ids_2(raw_graph, block_to_graph)
		block_to_graph.edata['_ID'] = current_block_eidx_raw
		if layer == 0:
			my_graph_partitioner=Graph_Partitioner(block_to_graph, args) #init a graph partitioner object
			batched_output_nid_list,weights_list,batch_list_generation_time, p_len_list=my_graph_partitioner.init_graph_partition()

			# print('partition_len_list')
			# print(p_len_list)
			args.batch_size=my_graph_partitioner.batch_size
			
			blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer(raw_graph, block_to_graph, batched_output_nid_list)
			# TODO
			#change the generate block
			connection_time, block_gen_time, mean_block_gen_time = time_1
			# batch_list_generation_time = t1 - tt
			time_2 = (connection_time, block_gen_time, mean_block_gen_time, batch_list_generation_time)
		else:
			return
	data_loader=[]
	# TODO
	return data_loader, weights_list, time_2

def gen_grouped_dst_list(prev_layer_blocks):
	post_dst=[]
	for block in prev_layer_blocks:
		src_nids = block.srcdata['_ID'].tolist()
		post_dst.append(src_nids)
	return post_dst # return next layer's dst nids(equals prev layer src nids)

def generate_dataloader_wo_gp_Pure_range(raw_graph, block_to_graph_list, args):
	data_loader=[]
	weights_list=[]
	num_batch=0
	blocks_list=[]
	final_dst_list =[]
	final_src_list=[]
	prev_layer_blocks=[]
	t_2_list=[]
	# prev_layer_src_list=[]
	# prev_layer_dst_list=[]
	# print('now we generate block from output to src direction, bottom up direction')
	l=len(block_to_graph_list)
	# the order of block_to_graph_list is bottom-up(the smallest block at first order)
	#b it means the graph partition starts 
	# from the output layer to the first layer input block graphs.
	for layer, block_to_graph in enumerate(block_to_graph_list):
		# print('The real block id is ', l-1-layer)
		# print('block_to_graph.srcdata')
		# print(block_to_graph.srcdata)
		dst_nids=block_to_graph.dstdata['_ID']
		src_nids=block_to_graph.srcdata['_ID'].tolist()
		eid_nids=block_to_graph.edata['_ID'].tolist()
		current_block_eidx_raw=eid_nids
		
		if layer ==0:
			batched_output_nid_list, weights_list=gen_batched_output_list(dst_nids, args.batch_size,'range')
			num_batch=len(batched_output_nid_list)
			# print('num of batch ',num_batch )
			# print('layer ', l-1-layer)
			
			# print('\tselection method range initialization spend ', time.time()-t1)
			# block 0 : (src_0, dst_0); block 1 : (src_1, dst_1);.......
			blocks, src_list, dst_list,time_1 = generate_blocks_for_one_layer(raw_graph, block_to_graph,  batched_output_nid_list)
			
			prev_layer_blocks=blocks
			blocks_list.append(blocks)
			final_dst_list=dst_list
			if layer==args.num_layers-1:
				final_src_list=src_list
		else:
			grouped_output_nid_list=gen_grouped_dst_list(prev_layer_blocks)
			num_batch=len(grouped_output_nid_list)
			# print('num of batch ',num_batch )
			
			blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer(raw_graph, block_to_graph, grouped_output_nid_list)
			
			if layer==args.num_layers-1: # if current block is the final block, the src list will be the final src
				final_src_list=src_list
			else:
				prev_layer_blocks=blocks
	
			blocks_list.append(blocks)

	
	for batch_id in range(num_batch):
		cur_blocks=[]
		for i in range(args.num_layers-1,-1,-1):
			cur_blocks.append(blocks_list[i][batch_id])
		
		dst = final_dst_list[batch_id]
		src = final_src_list[batch_id]
		data_loader.append((src, dst, cur_blocks))
	
	# sum_list=[]
	# if len(t_2_list)==1:
	# 	sum_list=t_2_list[0]
	# elif len(t_2_list)==2:
	# 	for bb in range(0,len(t_2_list),2):
	# 		list1=t_2_list[bb]
	# 		list2=t_2_list[bb+1]
	# 		for (item1, item2) in zip(list1, list2):
	# 			sum_list.append(item1+item2)

	# elif len(t_2_list)==3:
	# 	for bb in range(0,len(t_2_list),3):
	# 		list1=t_2_list[bb]
	# 		list2=t_2_list[bb+1]
	# 		list3=t_2_list[bb+2]
	# 		for (item1, item2, item3) in zip(list1, list2, list3):
	# 			sum_list.append(item1+item2+item3)

	return data_loader, weights_list, [[],[],0,[]]
	# return data_loader, weights_list, sum_list
		

def merge_2_a_list(ids_list):
	return sum(ids_list,[])

def to_tensor(list_):
	return torch.tensor(list_, dtype=torch.long)

def sort_list_(ids_list):
	ids=sum(ids_list,[])
	res=[]
	[res.append(i) for i in ids if i not in res]
	return res

def remove_duplicate(ids_list):
	res=[]
	[res.append(i) for i in ids_list if i not in res]
	return res
	

def gen_batched_output_list(dst_nids, batch_size, partition_method):
	batched_output_nid_list=[]
	weights_list=[]
	if partition_method=='range':
		indices = [i for i in range(len(dst_nids))]
		map_output_list = list(numpy.array(dst_nids)[indices])
		batches_nid_list = [map_output_list[i:i + batch_size] for i in range(0, len(map_output_list), batch_size)]
		length = len(dst_nids)
		weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]
	return batches_nid_list, weights_list

def Reverse(lst):
	import copy
	a=copy.deepcopy(lst)
	a.reverse()
	return a


def generate_dataloader_wo_gp_Pure_range_block(raw_graph, full_block_dataloader, args):
	data_loader=[]
	weights_list=[]
	num_batch=0
	blocks_list=[]
	final_dst_list =[]
	final_src_list=[]
	prev_layer_blocks=[]
	t_2_list=[]
	
	connect_checking_time_list=[]
	block_gen_time_total=0
	batch_blocks_gen_mean_time=0
	# print('now we generate block from output to src direction, bottom up direction')
	
	# the order of layer_block_list is bottom-up(the smallest block at first order)
	#b it means the graph partition starts 
	# from the output layer to the first layer input block graphs.
	for _,(src_full, dst_full, full_blocks) in enumerate(full_block_dataloader): # only one full batch blocks
		block_gen_time_total=0
		# reversed_blocks=Reverse(blocks)
		l=len(full_blocks)
		# for layer, layer_block in enumerate(reversed_blocks):
		for layer_id, layer_block in enumerate(reversed(full_blocks)):
			# print('layer id ', layer_id)
			# print('The real block id is ', l-1-layer_id)
		
			dst_nids=layer_block.dstdata['_ID']
			# src_nids=layer_block.srcdata['_ID'].tolist()
			# print('layer_block.edata[_ID].tolist()')
			# print(layer_block.edata['_ID'].tolist()) # eids 
			# 
			# print()
			bb=time.time()
			block_eidx_global, block_edges_nids_global = get_global_graph_edges_ids_block(raw_graph, layer_block)
			get_eid_time=time.time()-bb
			# print('get_global_graph_edges_ids_block function  spend '+ str(get_eid_time))
			# print('after block_eidx_global, block_edges_nids_global = get_global_graph_edges_ids_block(raw_graph, layer_block)')
			# print(sorted(block_eidx_global.tolist()))
			# print()

			layer_block.edata['_ID'] = block_eidx_global
			# eid_nids=layer_block.edata['_ID'].tolist() # global eids in this layer block
			if layer_id ==0:

				# block_eidx_global, block_edges_nids_global = get_global_graph_edges_ids_block(raw_graph, layer_block)
				# print('after block_eidx_global, block_edges_nids_global = get_global_graph_edges_ids_block(raw_graph, layer_block)')
				# # print(sorted(block_eidx_global.tolist()))
				# print()
				# layer_block.edata['_ID'] = block_eidx_global

				t1= time.time()
				batched_output_nid_list, weights_list=gen_batched_output_list(dst_nids, args.batch_size, args.selection_method)
				num_batch=len(batched_output_nid_list)
				# print('num of batch ',num_batch )
				# print('layer ', l-1-layer)
				select_time=time.time()-t1
				# print(str(args.selection_method)+' selection method range initialization spend '+ str(select_time))
				# block 0 : (src_0, dst_0); block 1 : (src_1, dst_1);.......
				blocks, src_list, dst_list,time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block,  batched_output_nid_list)
				# return
				prev_layer_blocks=blocks
				blocks_list.append(blocks)
				final_dst_list=dst_list
				if layer_id==args.num_layers-1:
					final_src_list=src_list
			else:
				grouped_output_nid_list=gen_grouped_dst_list(prev_layer_blocks)
				num_batch=len(grouped_output_nid_list)
				# print('num of batch ',num_batch )
				blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block, grouped_output_nid_list)

				if layer_id==args.num_layers-1: # if current block is the final block, the src list will be the final src
					final_src_list=src_list
				else:
					prev_layer_blocks=blocks
		
				blocks_list.append(blocks)
				connection_time, block_gen_time=time_1
				connect_checking_time_list.append(connection_time)
				block_gen_time_total+=block_gen_time
		# connect_checking_time_res=sum(connect_checking_time_list)
		batch_blocks_gen_mean_time= block_gen_time_total/num_batch
	
	
	for batch_id in range(num_batch):
		cur_blocks=[]
		for i in range(args.num_layers-1,-1,-1):
			cur_blocks.append(blocks_list[i][batch_id])
		
		dst = final_dst_list[batch_id]
		src = final_src_list[batch_id]
		data_loader.append((src, dst, cur_blocks))
	
	args.num_batch=num_batch
	
	return data_loader, weights_list, [sum(connect_checking_time_list), block_gen_time_total, batch_blocks_gen_mean_time]
	# return data_loader, weights_list, sum_list


def generate_dataloader_block(raw_graph, full_block_dataloader, args):

	if 'partition' in args.selection_method:
		return generate_dataloader_w_partition(raw_graph, block_to_graph_list, args)
	else:
		return generate_dataloader_wo_gp_Pure_range_block(raw_graph, full_block_dataloader, args)
		

# def generate_dataloader(raw_graph, block_to_graph_list, args):
	
# 	if 'partition' in args.selection_method:
# 		return generate_dataloader_w_partition(raw_graph, block_to_graph_list, args)
# 	else:
# 		return generate_dataloader_wo_gp_Pure_range(raw_graph, block_to_graph_list, args)
	


def print_reconstruct_blocks(raw_graph, cur_block_layer, compare_blocks, b_id):
		flag=True
		print()
		print('--------------------------------------------------------------------------------------------------------------')
		# print(block_layer.dstdata)
		# print(compare_blocks[l-1-iii].dstdata)
		# print('iii ', iii)
		print('Block (layer ) ', b_id)
		print('----->   re v.s. full')
		print('\t   <-----------------------       tht sorted version (dst, src, eid)     to check match or not --------------> ')
		print('*'*50)
		if sorted(cur_block_layer.dstdata['_ID'].tolist())!=sorted(compare_blocks.dstdata['_ID'].tolist()):
			print(sorted(cur_block_layer.dstdata['_ID'].tolist()))
			print(sorted(compare_blocks.dstdata['_ID'].tolist()))
			print('_ -'*30)
			flag=False
		else:
			print('dst sorted match.....')
			
		print()
		if sorted(cur_block_layer.srcdata['_ID'].tolist())!=sorted(compare_blocks.srcdata['_ID'].tolist()):
			print(sorted(cur_block_layer.srcdata['_ID'].tolist()))
			print(sorted(compare_blocks.srcdata['_ID'].tolist()))
			print('_ -'*30)
			flag=False
		else:
			print('src sorted match.....')
		print()
		ttmp, (_,_) =get_global_graph_edges_ids_block(raw_graph, compare_blocks)
		if sorted(cur_block_layer.edata['_ID'].tolist()) != sorted(ttmp.tolist()):
			print(sorted(cur_block_layer.edata['_ID'].tolist()))
			print(sorted(ttmp.tolist()))
			flag=False
		else:
			print('eids sorted match........')
		print()
		print('len eids global:',len(ttmp))
		print()
		# print(sorted(compare_blocks[l-1-iii].edata['_ID'].tolist()))
		# print('len eids local',len(compare_blocks[l-1-iii].edata['_ID']))
		
		# if flag==True:
		# 	return
		print()
		print('\t     <-------------    tht unsorted reconstruct block v.s. full batch         (dst, src, eid)  -----------------> ')
		print('reconstruct block edges()')
		print(cur_block_layer.edges())
		print('compare block edges()')
		print(compare_blocks.edges())
		print('reconstruct block nodes()')
		# print(cur_block_layer.nodes('_N_dst'))
		# print(cur_block_layer.nodes('_N_src'))
		# print('compare block nodes()')
		# print(compare_blocks.nodes())
		# print(induced_src[r].tolist())
		print('*'*50)
		print((cur_block_layer.dstdata['_ID'].tolist()))
		print((compare_blocks.dstdata['_ID'].tolist()))
		print('_ -'*30)
		print()
		print((cur_block_layer.srcdata['_ID'].tolist()))
		print((compare_blocks.srcdata['_ID'].tolist()))
		print('_ -'*30)
		print()
		print((cur_block_layer.edata['_ID'].tolist()))
		print()
		# print(sorted(compare_blocks[l-1-iii].edata['_ID'].tolist()))
		# print('len eids local',len(compare_blocks[l-1-iii].edata['_ID']))
		ttmp, (_,_) =get_global_graph_edges_ids_block(raw_graph, compare_blocks)
		print(sorted(ttmp.tolist()))
		print()
		print('len eids global:',len(ttmp))
		print('--------------------------------------------------------------------------------------------------------------')
	
	
def order_replace(cur_block_layer, compare_blocks):
	cur_block_layer.dstdata['_ID']=compare_blocks.dstdata['_ID']
	cur_block_layer.srcdata['_ID']=compare_blocks.srcdata['_ID']
	cur_block_layer.edata['_ID']=compare_blocks.edata['_ID']
	# print(cur_block_layer.ntypes)
	# print(compare_blocks.ntypes)
	# print(cur_block_layer.etypes)
	# print(compare_blocks.etypes)
	# if cur_block_layer.nodes('_N').tolist()!=compare_blocks.nodes('_N').tolist():
	# 	print('not match nodes')
	# 	print(cur_block_layer.nodes('_N'))
	# 	print(compare_blocks.nodes('_N'))
	# # cur_block_layer.nodes('_N')=compare_blocks.nodes('_N')
	# if cur_block_layer.edges(order='eid')[0].tolist()!=compare_blocks.edges(order='eid')[0].tolist():
	# 	print('edges not match ')
	# 	print(cur_block_layer.edges(order='eid')[0].tolist())
	# 	print(compare_blocks.edges(order='eid')[0].tolist())
	return cur_block_layer


def print_dataloader_info(raw_graph, full_batch_dataloader):
	print('full batch dataloader information ......................................')
	for _,(src_full, dst_full, blocks_full) in enumerate(full_batch_dataloader):
		src_, dst_, compare_blocks = src_full, dst_full, blocks_full
		print('the full batch nids ')
		print('-------------the full batch src------------')
		print(sorted(src_.tolist()))
		print('--------------the full batch dst-----------')
		print(sorted(dst_.tolist()))
		print('full batch blocks')
		for llayer, B in enumerate (compare_blocks):
			print('Block layer ', llayer)
			layer_src = B.srcdata[dgl.NID]
			layer_dst = B.dstdata[dgl.NID]
			layer_eid = B.edata[dgl.NID]
			layer_eid_global, (e_s,e_d) =get_global_graph_edges_ids_block(raw_graph, B)
			print(B)
			print()
			print('------------------layer src global-----------------------')
			print(layer_src.tolist()) 
			print()
			print('-----------------------layer dst global-----------------------')
			print(layer_dst.tolist())
			print()
			print('--------------------------layer_eid local-----------------------')
			print(layer_eid.tolist())
			print()
			print('--------------------------------sorted layer_eid_global-----------------------')
			print(sorted(layer_eid_global.tolist()))
			print()
			print('------print edges (src, dst, eid)----------')
			print_edges_global(raw_graph, B)
			print('~'*40)
			print()
		return compare_blocks


def reconstruct_subgraph_manually(raw_graph, block_dataloader, full_batch_dataloader ):
	"""
	this version consider the order of eids in reconstruct
	
	"""
	# compare_blocks=print_dataloader_info(raw_graph, full_batch_dataloader)
	print('=========================================================')
	print('reconstruct  information ......................................')
	#---------------------------------------------------------------------------
	layers=[[],[]]
	# layers=[[]]
	# layers=[[]] * len(full_batch_dataloader)
	res=[]
	re_src=[]
	re_dst=[]
	
	for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):
		for i, bi in enumerate(blocks):
			print('block ',i)
			print(bi)
			layers[i].append(bi)
	# collect block layer by layer
	print()
	l=len(layers)
	prev_layer_dst=[]
	prev_layer_src=[]
	prev_layer_eid=[]
	reversed_layers=Reverse(layers)
	for iii, layer in enumerate(reversed_layers):
		cur_layer_dst=[]
		cur_layer_src=[]		
		cur_layer_eids=[]
		for i, bi in enumerate(layer):
			cur_layer_dst.append(bi.dstdata[dgl.NID].tolist())
		
		for i, bi in enumerate(layer):
			cur_layer_src.append(bi.srcdata[dgl.NID].tolist())
			cur_layer_eids.append(bi.edata[dgl.NID].tolist())

		cur_layer_dst=sum(cur_layer_dst,[])
		# print(sorted(cur_layer_src.tolist()))
		cur_layer_dst=remove_duplicate(prev_layer_dst + cur_layer_dst)	
		
		cur_layer_src=sum(cur_layer_src,[])
		cur_layer_src=remove_duplicate(prev_layer_dst+prev_layer_src + cur_layer_src)	

		cur_layer_eids = sum(cur_layer_eids,[])
		cur_layer_eids=remove_duplicate(prev_layer_eid+cur_layer_eids)

		prev_layer_dst = cur_layer_dst
		prev_layer_src = cur_layer_src
		prev_layer_eid = cur_layer_eids

		cur_layer_src = to_tensor(cur_layer_src)
		cur_layer_dst = to_tensor(cur_layer_dst)
		cur_layer_eids = to_tensor(cur_layer_eids)

		if iii==0:	
			re_dst=cur_layer_dst
		if iii==l-1:
			re_src=cur_layer_src
		# 	prev_layer_dst=layer_dst
		# else: 
		# 	layer_dst=sort_list_(prev_layer_dst+sum(layer_dst,[])) # 
		
		layer_sub_graph = dgl.edge_subgraph(raw_graph, cur_layer_eids, store_ids=True) # layer_eids is global 
		# print(layer_sub_graph.ndata)
		# print('layer_sub_graph.ndata[_ID]')
		# print(sorted(layer_sub_graph.ndata['_ID']))
		# print('layer_sub_graph.edata[_ID]')
		# print(layer_sub_graph.edata['_ID'])
		# print(sorted(layer_sub_graph.edata['_ID'].tolist()))
		# # print('layer_sub_graph.ndata[train_mask]')
		# # print(layer_sub_graph.ndata['_ID'][layer_sub_graph.ndata['train_mask']])
		# print('layer_sub_graph.edges()[1]')
		# print(layer_sub_graph.edges()[1])
		tmp=layer_sub_graph.edges(order='eid')[1].tolist()
		r=[]
		[r.append(i) for i in tmp if i not in r]
		tmp_s=layer_sub_graph.edges(order='eid')[0].tolist()
		r_s=[]
		[r_s.append(i) for i in tmp_s if i not in r_s and i not in r]
		# print('-*'*40)
		# print('r_s   ', r_s)
		# print('-*'*40)
		cur_block_layer = dgl.to_block(layer_sub_graph, dst_nodes=r) # r is local dst nid
		b_induced_ndata = layer_sub_graph.ndata['_ID']
		cur_block_layer.dstdata['_ID']= b_induced_ndata[r]
		# print(" dst of this layer ")
		# print(sorted(b_induced_ndata[r].tolist()))
		cur_block_layer.srcdata['_ID']= b_induced_ndata[r+r_s]
		cur_block_layer.edata['_ID']=cur_layer_eids
		num_batch=len(block_dataloader)
		print('the number of batches : ', num_batch)
		# print('-------------------------------------------------------------------------------------------------')
		# print_reconstruct_blocks(raw_graph, cur_block_layer, compare_blocks[l-1-iii], l-1-iii)
		# order_replace(cur_block_layer, compare_blocks[l-1-iii])
		# print('--------------------------------------------------------------------------------------------------')
		# res.append(cur_block_layer)
		res.insert(0, cur_block_layer)
	# res=(re_src, re_dst, res)
	return [(re_src, re_dst, res)]



# not change order of eid
def reconstruct_subgraph(raw_graph, block_dataloader, full_batch_dataloader ):
	
	# compare_blocks=print_dataloader_info(raw_graph, full_batch_dataloader)
	print('=========================================================')
	print('reconstruct  information ......................................')
	#---------------------------------------------------------------------------
	layers=[[],[]]
	# layers=[[]]
	# layers=[[]] * len(full_batch_dataloader)
	res=[]
	re_src=[]
	re_dst=[]
	
	for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):
		for i, bi in enumerate(blocks):
			print('block ',i)
			print(bi)
			layers[i].append(bi)
	# collect block layer by layer
	print()
	l=len(layers)
	prev_layer_dst=[]
	prev_layer_eid=[]
	reversed_layers=Reverse(layers)
	for iii, layer in enumerate(reversed_layers):
		cur_layer_dst=[]
		cur_layer_src=[]		
		cur_layer_eids=[]
		for i, bi in enumerate(layer):
			cur_layer_dst.append(bi.dstdata[dgl.NID].tolist())
		
		for i, bi in enumerate(layer):
			cur_layer_src.append(bi.srcdata[dgl.NID].tolist())
			cur_layer_eids.append(bi.edata[dgl.NID].tolist())

		cur_layer_dst=sort_list_(cur_layer_dst)				
		cur_layer_src=sum(cur_layer_src,[])
		cur_layer_src=torch.tensor(remove_duplicate(cur_layer_dst + cur_layer_src),dtype=torch.long)	

		
		# print(sorted(cur_layer_src.tolist()))
		cur_layer_dst=torch.tensor(cur_layer_dst,dtype=torch.long)	
		cur_layer_eids=torch.tensor(sort_list_(cur_layer_eids),dtype=torch.long)
		if iii==0:	
			re_dst=cur_layer_dst
		if iii==l-1:
			re_src=cur_layer_src
		# 	prev_layer_dst=layer_dst
		# else: 
		# 	layer_dst=sort_list_(prev_layer_dst+sum(layer_dst,[])) # 
		
		layer_sub_graph=dgl.edge_subgraph(raw_graph, cur_layer_eids, store_ids=True) # layer_eids is global 
		# print(layer_sub_graph.ndata)
		# print('layer_sub_graph.ndata[_ID]')
		# print(sorted(layer_sub_graph.ndata['_ID']))
		# print('layer_sub_graph.edata[_ID]')
		# print(layer_sub_graph.edata['_ID'])
		# print(sorted(layer_sub_graph.edata['_ID'].tolist()))
		# # print('layer_sub_graph.ndata[train_mask]')
		# # print(layer_sub_graph.ndata['_ID'][layer_sub_graph.ndata['train_mask']])
		# print('layer_sub_graph.edges()[1]')
		# print(layer_sub_graph.edges()[1])
		tmp=layer_sub_graph.edges(order='eid')[1].tolist()
		r=[]
		[r.append(i) for i in tmp if i not in r]
		tmp_s=layer_sub_graph.edges(order='eid')[0].tolist()
		r_s=[]
		[r_s.append(i) for i in tmp_s if i not in r_s and i not in r]
		# print('-*'*40)
		# print('r_s   ', r_s)
		# print('-*'*40)
		cur_block_layer = dgl.to_block(layer_sub_graph, dst_nodes=r) # r is local dst nid
		b_induced_ndata = layer_sub_graph.ndata['_ID']
		cur_block_layer.dstdata['_ID']= b_induced_ndata[r]
		# print(" dst of this layer ")
		# print(sorted(b_induced_ndata[r].tolist()))
		cur_block_layer.srcdata['_ID']= b_induced_ndata[r+r_s]
		cur_block_layer.edata['_ID']=cur_layer_eids
		num_batch=len(block_dataloader)
		print('the number of batches : ', num_batch)
		# print('-------------------------------------------------------------------------------------------------')
		# print_reconstruct_blocks(raw_graph, cur_block_layer, compare_blocks[l-1-iii], l-1-iii)
		# order_replace(cur_block_layer, compare_blocks[l-1-iii])
		# print('--------------------------------------------------------------------------------------------------')
		# res.append(cur_block_layer)
		res.insert(0, cur_block_layer)
	# res=(re_src, re_dst, res)
	return [(re_src, re_dst, res)]

def _reconstruct_pickle(obj):
	# import io
	f = io.BytesIO()
	f.open('filename.pickle', 'wb')
	pickle.dump(obj, f)
	f.close()
	# f.write((b"randomContent")
	# f.name = "someFilename.pickle"
	f.seek(0)
	f = io.BytesIO()
	f.open('filename.pickle', 'rb')
	obj = pickle.load(f)
	f.close()

	return obj

def print_edges(cur_block):
	induced_src = cur_block.srcdata[dgl.NID]
	induced_dst = cur_block.dstdata[dgl.NID]
	induced_eid = cur_block.edata[dgl.NID].tolist()
	
	e_src_local, e_dst_local = cur_block.edges(order='eid')
	e_src, e_dst = induced_src[e_src_local], induced_src[e_dst_local]
	e_src = e_src.detach().numpy().astype(int)
	e_dst = e_dst.detach().numpy().astype(int)
	
	combination = [p for p in zip(e_src, e_dst, induced_eid)]
	print(combination)


def print_edges_global(raw_graph,cur_block):
	induced_src = cur_block.srcdata[dgl.NID]
	induced_dst = cur_block.dstdata[dgl.NID]
	induced_eid = cur_block.edata[dgl.NID].tolist()
	
	e_src_local, e_dst_local = cur_block.edges(order='eid')
	e_src, e_dst = induced_src[e_src_local], induced_src[e_dst_local]
	e_src = e_src.detach().numpy().astype(int)
	e_dst = e_dst.detach().numpy().astype(int)
	
	eid_global, (e_s,e_d) =get_global_graph_edges_ids_block(raw_graph, cur_block)
	# if torch.is_tensor(eid_global):
	# 	eid_global=eid_global.tolist()
	eid_global = eid_global.detach().numpy().astype(int)
	

	combination = [p for p in zip(e_src, e_dst, eid_global)]
	print(combination)


def check_graph_equal(g1, g2, *, check_idtype=True, check_feature=True):
	assert g1.device == g1.device
	if check_idtype:
		assert g1.idtype == g2.idtype
	assert g1.ntypes == g2.ntypes
	assert g1.etypes == g2.etypes
	assert g1.srctypes == g2.srctypes
	assert g1.dsttypes == g2.dsttypes
	assert g1.canonical_etypes == g2.canonical_etypes
	assert g1.batch_size == g2.batch_size

	# check if two metagraphs are identical
	for edges, features in g1.metagraph().edges(keys=True).items():
		assert g2.metagraph().edges(keys=True)[edges] == features
	for nty in g1.ntypes:
		assert g1.number_of_nodes(nty) == g2.number_of_nodes(nty)
		# assert F.allclose(g1.batch_num_nodes(nty), g2.batch_num_nodes(nty))
	for ety in g1.canonical_etypes:
		assert g1.number_of_edges(ety) == g2.number_of_edges(ety)
		# assert F.allclose(g1.batch_num_edges(ety), g2.batch_num_edges(ety))
		# src1, dst1, eid1 = g1.edges(etype=ety, form='all')
		# src2, dst2, eid2 = g2.edges(etype=ety, form='all')
