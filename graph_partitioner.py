import numpy 
import dgl
from numpy.core.numeric import Infinity
import multiprocessing as mp
import torch
import time
from statistics import mean
from my_utils import *

block_to_graph=None
# batches_nid_list_=[]



def InitializeBitDict(A_o, B_o):
    
    bit_dict=dict.fromkeys(A_o,0)
    bit_dict.update(dict.fromkeys(B_o,1))
    # for k in A:
    #     bit_dict[k] = 0
    # for m in B:
    #     bit_dict[m] = 1  
    return bit_dict


def get_two_partition_seeds(bit_dict):
    
    A_o=[k for k in bit_dict if bit_dict[k] == 0]
    B_o=[k for k in bit_dict if bit_dict[k] == 1]
    
    return  A_o, B_o

    
def calculate_redundancy( idx,  i, A_o, B_o, side, locked_nodes):
    global block_to_graph
    gain=0
    in_nids=block_to_graph.predecessors(i).tolist()
    # print('in_nids')
    # print(in_nids)

    if side==0:
        gain_pos=len(list(set(in_nids).intersection(set(B_o))))
        gain_neg=len(list(set(in_nids).intersection(set(A_o)))) 
    else:
        gain_pos=len(list(set(in_nids).intersection(set(A_o))))
        gain_neg=len(list(set(in_nids).intersection(set(B_o)))) 

    gain=gain_pos-gain_neg 
    # print('gain ',gain)
    if gain>=0 and not locked_nodes[i]:
        return (idx,i)
    
    return (idx,None)


def parallel_gen_batch_list(idx, nid, batches_nid_list_):
    # global batches_nid_list_
    global block_to_graph

    # p_len_list=get_partition_src_len_list()
    partition_src_len_list=[]
    for seeds in batches_nid_list_:
        in_ids=list(block_to_graph.in_edges(seeds))[0].tolist()
        src_len= len(list(set(in_ids+seeds)))
        partition_src_len_list.append(src_len)
        
    min_len=min(partition_src_len_list)
    P_ID= partition_src_len_list.index(min_len)
    print('p_id ', P_ID)

    
    # batches_nid_list_[P_ID].append(nid)
    return (idx, P_ID, nid )



class Graph_Partitioner:
    def __init__(self, layer_block, args):
        self.balanced_init_ratio=args.balanced_init_ratio
        self.dataset=args.dataset
        self.layer_block=layer_block # local graph with global nodes indices
        self.local=False
        self.output_nids=layer_block.dstdata['_ID'] # tensor type
        self.local_output_nids=[]
        self.src_nids_list= layer_block.srcdata['_ID'].tolist()
        self.full_src_len=len(layer_block.srcdata['_ID'])
        self.global_batched_seeds_list=[]
        self.local_batched_seeds_list=[]
        self.weights_list=[]
        self.alpha=args.alpha 
        self.walkterm=args.walkterm
        self.num_batch=args.num_batch
        self.selection_method=args.selection_method
        self.batch_size=0
        self.ideal_partition_size=0

        self.bit_dict={}
        self.side=0
        self.partition_nodes_list=[]
        self.partition_len_list=[]

        self.time_dict={}
        self.red_before=[]
        self.red_after=[]
        self.args=args
        

    # def global_to_local(self):
        
    #     sub_in_nids = self.src_nids_list
    #     global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
    #     local_batched_seeds_list=[]
    #     for global_in_nids in self.global_batched_seeds_list:
    #         local_in_nids = list(map(global_nid_2_local.get, global_in_nids))
    #         local_batched_seeds_list.append(local_in_nids)
        
    #     self.local_batched_seeds_list=local_batched_seeds_list
    #     self.local=True
    #     return 

    def global_to_local(self):
        
        sub_in_nids = self.src_nids_list
        global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
        self.local_output_nids = list(map(global_nid_2_local.get, self.output_nids.tolist()))
        
        
        # local_batched_seeds_list=[]
        # for global_nids in self.global_batched_seeds_list:
        #     local_nids = list(map(global_nid_2_local.get, global_nids))
        #     local_batched_seeds_list.append(local_nids)
        
        # self.local_batched_seeds_list=local_batched_seeds_list
        self.local=True
        return 
        
    
    def local_to_global(self):
        sub_in_nids = self.src_nids_list
        local_nid_2_global = { i: sub_in_nids[i] for i in range(0, len(sub_in_nids))}
        
        global_batched_seeds_list=[]
        for local_in_nids in self.local_batched_seeds_list:
            global_in_nids = list(map(local_nid_2_global.get, local_in_nids))
            global_batched_seeds_list.append(global_in_nids)

        self.global_batched_seeds_list=global_batched_seeds_list
        self.local=False
        return 

    def update_partition_size(self, nids, P_ID):
        self.local_batched_seeds_list[P_ID] += nids
        p_len_list=self.get_partition_src_len_list()
        self.partition_len_list=p_len_list
        min_len=min(p_len_list)
        p_id= p_len_list.index(min_len)

        return
    



    def get_the_smallest_partition_id(self,batches_nid_list_):
        # global batches_nid_list_
        if self.num_batch!=len(batches_nid_list_):
            print('get the smallest partition id error')
            return 
        self.local_batched_seeds_list=batches_nid_list_
        p_len_list=self.get_partition_src_len_list()
        self.partition_len_list=p_len_list
        min_len=min(p_len_list)
        p_id= p_len_list.index(min_len)

        # print('p_id '+str(p_id)+' '+str(min_len))
        
        return p_id

    def balanced_init(self):
        
        # batches_nid_list=[[] for i in range(NB)]
        nids=self.local_output_nids
        output_num=len(self.local_output_nids)
        
        indices = [i for i in range(len(nids))]
        map_output_list = list(numpy.array(nids)[indices])
        
        start=0
        mini_batch=int(self.batch_size*self.balanced_init_ratio)
        if mini_batch*self.num_batch<len(self.local_output_nids):
            start=mini_batch*self.num_batch
        else: 
            start=(mini_batch-1)*self.num_batch

        batches_nid_list = [map_output_list[i:i + mini_batch] for i in range(0, start, mini_batch)]
        idx=start
        if self.dataset=='karate':
            while idx < len(nids):
                P_ID=self.get_the_smallest_partition_id(batches_nid_list)
                batches_nid_list[P_ID].append(idx)
                idx+=1
                if idx%50==0:
                    print('p_id ', P_ID)
        else:

            # step = int(len(self.local_output_nids)*(1-self.balanced_init_ratio)/self.num_batch)-2
            # for idx in range(start,len(nids),step):
            #     P_ID=self.get_the_smallest_partition_id(batches_nid_list)
            #     batches_nid_list[P_ID]+=nids[idx:idx+step]
            #     # if idx%50==0:
            #     #     print('p_id ', P_ID)
            #     print('p_id ', P_ID)
            for idx in range(start,len(nids)):
                P_ID=self.get_the_smallest_partition_id(batches_nid_list)
                batches_nid_list[P_ID].append(idx)
                if idx%50==0:
                    print('p_id ', P_ID)
                
            
        
        

        weights_list = []
        for batch_nids in batches_nid_list:
            # temp = len(i)/output_num
            weights_list.append(len(batch_nids)/output_num)

        # self.local_batched_seeds_list=batches_nid_list
        # self.weights_list=weights_list
        return batches_nid_list,weights_list


    def gen_batched_seeds_list(self):
        '''

        Parameters
        ----------
        OUTPUT_NID: final layer output nodes id (tensor)
        selection_method: the graph partition method

        Returns
        -------

        '''
	
        full_len = len(self.local_output_nids)  # get the total number of output nodes
        self.batch_size=get_mini_batch_size(full_len,self.num_batch)
        
        indices=[]
        if self.selection_method == 'range_init_graph_partition' :
            t=time.time()
            indices = [i for i in range(full_len)]
            batches_nid_list, weights_list=gen_batch_output_list(self.local_output_nids,indices,self.batch_size)
            print('range_init for graph_partition spend: ', time.time()-t)
        elif self.selection_method == 'random_init_graph_partition' :
            t=time.time()
            indices = random_shuffle(full_len)
            batches_nid_list, weights_list=gen_batch_output_list(self.local_output_nids,indices,self.batch_size)
            print('random_init for graph_partition spend: ', time.time()-t)
        elif self.selection_method == 'balanced_init_graph_partition' :
            t=time.time()
            batches_nid_list, weights_list=self.balanced_init()
            print('balanced_init for graph_partition spend: ', time.time()-t)

            
        else:# selection_method == 'similarity_init_graph_partition':
            indices = torch.tensor(range(full_len)) #----------------------------TO DO
        
        # batches_nid_list, weights_list=gen_batch_output_list(self.output_nids,indices,self.batch_size)

        self.local_batched_seeds_list=batches_nid_list
        self.weights_list=weights_list

        # print('The batched output nid list before graph partition')
        # print_len_of_batched_seeds_list(batches_nid_list)

        return 


    def get_in_nodes(self,seeds):
        in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
        in_ids= list(set(in_ids))
        return in_ids


    def get_src_nodes(self,seeds):
        in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
        src_nids= list(set(in_ids+seeds))
        return src_nids


    def get_src_len(self,seeds):
        in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
        src_len= len(list(set(in_ids+seeds)))
        return src_len


    def get_partition_src_len_list(self):
        partition_src_len_list=[]
        for seeds_nids in self.local_batched_seeds_list:
            partition_src_len_list.append(self.get_src_len(seeds_nids))
        
        self.partition_src_len_list=partition_src_len_list
        return partition_src_len_list


    def get_redundancy_list(self):
        redundancy_list=[]
        for seeds_nids in self.local_batched_seeds_list:
            redundancy_list.append(self.get_src_len(seeds_nids)/self.ideal_partition_size)
        return redundancy_list


    def update_Batched_Seeds_list(self, batched_seeds_list, bit_dict, i, j):

        A_o=[k for k in bit_dict if bit_dict[k]==0]
        B_o=[k for k in bit_dict if bit_dict[k]==1]
        print(self.side)

        if self.get_src_len(A_o)<=self.get_src_len(B_o):
            print('side is 1')
            batch_i=A_o
            batch_j=B_o
        else:
            print('side is 0')
            batch_i=B_o
            batch_j=A_o

        batched_seeds_list.remove(batched_seeds_list[i])
        batched_seeds_list.insert(i,batch_i)
        
        batched_seeds_list.remove(batched_seeds_list[j])
        batched_seeds_list.insert(j,batch_j)

        return batched_seeds_list


    def getRedundancyRate(self, len_A, len_B):
    
        ratio_A = len_A/self.ideal_partition_size
        ratio_B = len_B/self.ideal_partition_size
        rate = (ratio_A + ratio_B)/2

        return rate, ratio_A, ratio_B 
    
    def balance_check_and_exchange_side(self):
        bit_dict=self.bit_dict

        A_o=[k for k in bit_dict if bit_dict[k] == 0]
        B_o=[k for k in bit_dict if bit_dict[k] == 1]
        
        len_A_part= self.get_src_len(A_o)
        len_B_part= self.get_src_len(B_o)
        len_total = len_A_part + len_B_part

        # if left partition size is less than right partition size, exchange side;
        # otherwise, do not exchange
        if len_B_part>0 and len_A_part>0 :
            # if len_A_part-len_B_part < len_total*self.alpha*0.1:
            if len_A_part-len_B_part >= 0:
                self.side=0
            else:
                self.side=1
                # bit_dict={i: 1-bit_dict[i] for i in bit_dict}
        else:
            print('there is a partition has no any nodes, error!!!!')

        # self.bit_dict=bit_dict
        return self.side


    def move_group_nids_balance_redundancy_check(self, ready_to_move_nids, red_rate):
        
        nids=ready_to_move_nids
        bit_dict=self.bit_dict

        for nid in nids:
            bit_dict[nid]=1-self.side
            
        A_o=[k for k in bit_dict if bit_dict[k] == 0]
        B_o=[k for k in bit_dict if bit_dict[k] == 1]
        
        
        balance_flag=False
        red_flag=False

        # t1=time.time()
        len_A_part = self.get_src_len(A_o)
        len_B_part = self.get_src_len(B_o)
        # print('get_src_nodes_len ', time.time()-t1)
        len_=len_A_part+len_B_part
        avg = len_/2
        
        if len_B_part>0 and len_A_part>0 and abs(len_A_part-len_B_part) < avg*self.alpha:
            balance_flag=True
        else:
            balance_flag=False

        
        # balance_flag=True    # now test 
        move_ratio, ratio_A, ratio_B=self.getRedundancyRate(len_A_part, len_B_part)
        
        
        if move_ratio < red_rate and ratio_A>=1 and ratio_B>=1:
            red_flag=True
            red_rate=move_ratio

        # red_flag=True if red<red_rate else False
        # if red_flag:
        #     print(' redundancy '+str(move_ratio))

        for nid in nids:
            bit_dict[nid]=1-bit_dict[nid]
        self.bit_dict=bit_dict

        move_flag = balance_flag and red_flag

        return move_flag, red_rate

    def get_Red_Rate(self): 

        A_o,B_o=get_two_partition_seeds(self.bit_dict)
        len_A_part= self.get_src_len(A_o)
        len_B_part= self.get_src_len(B_o)
        
        ratio_A = len_A_part/self.ideal_partition_size
        ratio_B = len_B_part/self.ideal_partition_size
        # cost = max(ratio_A,ratio_B)
        rate = mean([ratio_A,ratio_B])
        return rate, ratio_A,ratio_B 

    def update_Red_group(self, prev_ratio):
        
        #------------------------
        # bit_dict=balance_check_and_exchange(bit_dict)
        #-------------------------
        ready_to_move=[]

        A_o,B_o=get_two_partition_seeds(self.bit_dict)
        if self.side==0:
            output=A_o
        else:
            output=B_o

        if len(output)<=1:
            return

        # if len(output)>1: 
        if self.dataset=='karate':
            
            for idx, nid in enumerate(output):
                gain=0
                in_nids=self.layer_block.predecessors(nid).tolist()
                # print('nid \t', nid)
                # print('in_nids\t', in_nids)
                if self.side==0:
                    gain_pos=len(list(set(in_nids).intersection(set(B_o))))
                    gain_neg=len(list(set(in_nids).intersection(set(A_o)))) 
                else:
                    gain_pos=len(list(set(in_nids).intersection(set(A_o))))
                    gain_neg=len(list(set(in_nids).intersection(set(B_o)))) 

                gain=gain_pos-gain_neg 
                # print('gain \t',gain)
                # print()
                if gain>=0 and not self.locked_nodes[nid] :
                    ready_to_move.append(nid)
            # print(ready_to_move)

            while len(ready_to_move)>1:
                move_flag, rate_after_move = self.move_group_nids_balance_redundancy_check(ready_to_move, prev_ratio)
                if not move_flag :
                    # print('------------------------------- !!!!!!!!!!!!!! redundancy check failed, ')
                    gold_r=int(len(ready_to_move)*0.5)
                    ready_to_move=ready_to_move[:gold_r]
                else:
                    diff=prev_ratio-rate_after_move
                    # if diff>0:
                    print('\t\t\t redundancy will reduce ',diff)
                    break

        else: #args.dataset=='reddit' or other
            
            pool = mp.Pool(mp.cpu_count())
            tmp_gains = pool.starmap_async(calculate_redundancy, [(idx, i, A_o, B_o, self.side, self.locked_nodes) for idx, i in enumerate(output)]).get()
            pool.close()
            ready_to_move = [list(r)[1] for r in tmp_gains]
            ready_to_move=list(filter(lambda v: v is not None, ready_to_move))

            while len(ready_to_move)>1:    
                move_flag, rate_after_move = self.move_group_nids_balance_redundancy_check(ready_to_move, prev_ratio)
                if not move_flag :
                    # print('------------------------------- !!!!!!!!!!!!!! redundancy check failed, ')
                    gold_r=int(len(ready_to_move)*0.2)
                    ready_to_move=ready_to_move[:gold_r]
                else:
                    diff=prev_ratio-rate_after_move
                    # if diff>0:
                    print('\t\t\t redundancy will reduce ',diff)
                    break

        if len(output)==len(ready_to_move):
            ready_to_move=ready_to_move[:len(output)-1]    

        print('\t\t\t the number of node to move is :', len(ready_to_move))

        for i in ready_to_move:
            self.locked_nodes[i]=True
        for nid in ready_to_move:
            self.bit_dict[nid]=1-self.side
    
        
        return             
                    
                
    
    


    def walk_terminate_1(self,red_rate, update_times):
        
        bestRate =red_rate
        best_bit_dict=self.bit_dict
        print('\t walk terminate 1 start-------')
        
        side=self.balance_check_and_exchange_side() 
        
        # if left partition size is smaller than right partition, exchange them
        # make sure the left partition size is larger or equal with right partition size.
        # then, we can only focus the first step move nodes from the larger partition to the smaller one.
        A_o,B_o= get_two_partition_seeds(self.bit_dict) # get A_o, B_o based on global variable bit_dict
        subgraph_o = A_o+B_o
        locked_nodes={id:False for id in subgraph_o}
        self.locked_nodes=locked_nodes

        # update_times=5
        
        # t_b=time.time()
        for i in range(update_times):
            # tt=time.time()
            print('\t\t\t\t\t\t current side ',self.side)
            self.update_Red_group(red_rate)
            # tt_e=time.time()
    
            # print('\t\t\tone update redundancy spend time _*_\t', tt_e-tt)	
            print('\t\t\t --group redundancy rate update  step :'+ str(i)+'  side '+str(self.side))
            ratio,ratio_A,ratio_B=self.get_Red_Rate()
            print('\t\t\t redundancy rate (ration_mean, ratio_A, ratio_B): '+str(ratio)+',  '+str(ratio_A)+',  '+ str(ratio_B))
            if ratio < bestRate: 
                bestRate = ratio
                best_bit_dict = self.bit_dict

            self.side=1-self.side
        # t_e=time.time()
        # print('\t\tupdate redundancy of segment  ', t_e-t_b)


        if (bestRate < red_rate) : #is there improvement? Yes
            self.bit_dict = best_bit_dict
            return True, bestRate, best_bit_dict

        #is there improvement? No
        ratio,ratio_A,ratio_B=self.get_Red_Rate()
        if ratio_A > ratio_B:
            self.side=1
        else:
            self.side=0

        return False, red_rate, self.bit_dict
    

    def graph_partition_variant(self):
        # print(self.layer_block)
        # print(self.layer_block.edges())

        
        full_batch_subgraph=self.layer_block #heterogeneous graph
        self.bit_dict={}
        print('---------------------------- variant graph partition start---------------------')
        # if num_batch == 0:
        #     self.output_nids/
        self.ideal_partition_size=(self.full_src_len/self.num_batch)
    
        src_ids=list(full_batch_subgraph.edges())[0]
        dst_ids=list(full_batch_subgraph.edges())[1]
        local_g = dgl.graph((src_ids, dst_ids)) #homogeneous graph
        local_g = dgl.remove_self_loop(local_g)
        # from draw_graph import draw_graph
        # draw_graph(local_g)
        self.layer_block=local_g
        global block_to_graph
        block_to_graph=local_g

        self.gen_batched_seeds_list() # based on user choice to split output nodes 

        src_len_list= self.get_partition_src_len_list()

        print('before graph partition ')
        print_len_of_partition_list(src_len_list) 
        print('{}-'*40)
        print()

        
        for i in range(self.num_batch-1):# no (end, head) pair
            print('-------------------------------------------------------------  compare batch pair  (' +str(i)+','+str(i+1)+')')
            print_len_of_batched_seeds_list(self.local_batched_seeds_list) 
            tii=time.time()
            
            A_o=self.local_batched_seeds_list[i]
            B_o=self.local_batched_seeds_list[i+1]
            len_part_A = self.get_src_len(A_o)
            len_part_B = self.get_src_len(B_o)

            tij=time.time()
            print('\n\tpreparing two sides time : ' , time.time()-tii)

            self.bit_dict=InitializeBitDict(A_o,B_o)

            print('\tInitialize BitList time : ' , time.time()-tij)
            tik=time.time()

            red_rate,ratio_A,ratio_B =self.getRedundancyRate(len_part_A,len_part_B) #r_cost=max(r_A, r_B)
            
            print('\tgetRedundancyCost: time  ' , time.time()-tik)
            print()
            
            print('\t\t\t\t\tlength of partitions '+ str(len_part_A)+', '+str(len_part_B))
            print()


            self.red_before.append(ratio_A)
            if i==(self.num_batch-2):
                self.red_before.append(ratio_B)

            if red_rate <= 1.0:
                continue
            
            print('\tbefore terminate 1 the average redundancy rate is: ', red_rate)
            print('\t'+'-'*80)
            # walks=5
            walks=self.args.walks
            for walk_step in range(walks):
                if self.walkterm==1:
                    ti=time.time()

                    improvement,red_rate_after,bit_dict_after=self.walk_terminate_1(red_rate, self.args.update_times)
                    
                    print('\twalk terminate 1 spend time', time.time()-ti)
                    print('\t\t\t\t improvement: ',improvement)
                    # print('\t\t\tthe  redundancy rate ',red_rate_after)
                    
                    if not improvement or walk_step==walks-1: # go several walks steps, until there is no improvement, 

                        self.local_batched_seeds_list = self.update_Batched_Seeds_list(self.local_batched_seeds_list, self.bit_dict, i, i+1)
                        # update batched_seeds_list based on bit_dict
                        print('\t walk step '+ str(walk_step)+'  partition ')
                        src_len_list = self.get_partition_src_len_list()
                        print_len_of_partition_list(src_len_list) 
                        # print out the result after exchange batched_seeds_list
                        
                        print()
                        break
                
            #--------------------- initialization checking done   ----------------   
            print('\t'+'-'*50 +'end of batch '+ str(i))
        
    
        weight_list=get_weight_list(self.local_batched_seeds_list)
        src_len_list=self.get_partition_src_len_list()
        print('after graph partition')
        self.weights_list=weight_list
        self.partition_len_list=src_len_list

        return self.local_batched_seeds_list, weight_list, src_len_list
    
        
        # def get_time(function_name, params):
        #     t_start=time.time()
        #     _run_(function_name,params)
        #     t_end=time.time()
        #     tt= t_end-t_start
        #     print('the time spend on '+str(function_name)+' is '+str(tt))


    def init_graph_partition(self):
        ts = time.time()
        
        self.global_to_local() # global to local            self.local_batched_seeds_list
        print('global_2_local', (time.time()-ts))

        t1 = time.time()
        # self.gen_batched_seeds_list()

        t2=time.time()
        # Then, the graph_parition is run in block to graph local nids,it has no relationship with raw graph
        self.graph_partition_variant()
        print('graph partition algorithm spend time', time.time()-t2)
        # after that, we transfer the nids of batched output nodes from local to global.
        self.local_to_global() # local to global         self.global_batched_seeds_list
        t_total=time.time()-ts

        return self.global_batched_seeds_list, self.weights_list, t_total, self.partition_len_list

        

    def balanced_init_graph_partition(self):
        ts = time.time()
        
        
        self.global_to_local() # global to local            self.local_batched_seeds_list
        print('global_2_local', (time.time()-ts))
        t1 = time.time()
        
        t2=time.time()
        # Then, the graph_parition is run in block to graph local nids,it has no relationship with raw graph
        self.graph_partition_variant()
        print('graph partition algorithm spend time', time.time()-t2)
        # after that, we transfer the nids of batched output nodes from local to global.
        self.local_to_global() # local to global         self.global_batched_seeds_list
        t_total=time.time()-ts

        return self.global_batched_seeds_list, self.weights_list, t_total, self.partition_len_list























