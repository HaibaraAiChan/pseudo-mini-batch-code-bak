import os
import time
import numpy
import dgl

def parse_results(output: str):
    lines = output.split("\n")
    epoch_times = []
    final_train_acc = ""
    final_test_acc = ""
    for line in lines:
        line = line.strip()
        if line.startswith("Training time/epoch"):
            epoch_times.append(float(line.split(' ')[-1]))
        if line.startswith("Final Train"):
            final_train_acc = line.split(":")[-1]
        if line.startswith("Final Test"):
            final_test_acc = line.split(":")[-1]
    return {"epoch_time": np.array(epoch_times)[-10:].mean(),
            "final_train_acc": final_train_acc,
            "final_test_acc": final_test_acc}

def gen_batch_output_list(OUTPUT_NID,indices,mini_batch):

    map_output_list = list(numpy.array(OUTPUT_NID)[indices])
        
    batches_nid_list = [map_output_list[i:i + mini_batch] for i in range(0, len(map_output_list), mini_batch)]
            
    output_num = len(OUTPUT_NID)
    
    # print(batches_nid_list)
    weights_list = []
    for batch_nids in batches_nid_list:
        # temp = len(i)/output_num
        weights_list.append(len(batch_nids)/output_num)
        
    return batches_nid_list, weights_list 

def print_len_of_batched_seeds_list(batched_seeds_list):
    
    node_or_len=1   # print length of each batch
    print_list(batched_seeds_list,node_or_len) 
    return


def print_len_of_partition_list(partition_src_list_len):    
    
    print_len_list(partition_src_list_len)
    return

def print_list(nids_list, node_or_len):
    res=''
    if node_or_len==0:
        # print nodes_list
        for nids in nids_list:
            res=res+str(nids)+', '
        print('\t\t\t\t list :')
    
    else:
        for nids in nids_list:
            res=res+str(len(nids))+', '
            
        print('\t\t\t\t list len:')
    
    print('\t\t\t\t'+res)
    print()
    return


def print_len_list(nids_list):
    res=''
    
    for nids in nids_list:
        res=res+str(nids)+', '
    # print('\t\t\t\t list len : ')

    print('\t\t'+res)
    print()
    return

def random_shuffle(len):
	indices = numpy.arange(len)
	numpy.random.shuffle(indices)
	return indices

def get_mini_batch_size(full_len,num_batch):
	mini_batch=int(full_len/num_batch)
	if full_len%num_batch>0:
		mini_batch+=1
	# print('current mini batch size of output nodes ', mini_batch)
	return mini_batch

    
def get_weight_list(batched_seeds_list):
    
    output_num = len(sum(batched_seeds_list,[]))
    # print(output_num)
    weights_list = []
    for seeds in batched_seeds_list:
		# temp = len(i)/output_num
        weights_list.append(len(seeds)/output_num)
    return weights_list

