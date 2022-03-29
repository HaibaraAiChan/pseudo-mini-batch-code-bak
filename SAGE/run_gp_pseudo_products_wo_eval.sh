#!/bin/bash

folderPath=/home/cc/graph_partition_multi_layers/pseudo_mini_batch_full_batch/SAGE

File=${folderPath}/pseudo_mini_batch_products_sage.py
Data=ogbn-products


model=sage
seed=1236 
setseed=True
GPUmem=True
lr=0.01
dropout=0.5
weightDecay=5e-4
# layers=3
# hidden=256
run=1
epoch=2
# pMethodList=(random) 
# AggreList=(mean)

# batch_size=(60 30 15 8 4)
# num_batches=(2 8 32 128 256 4 16 64)
# num_batches=(2 8 32 128 256 4 16 64)
batch_size=(98308 49154 24577 12289 6145 3073 1537 769)
hiddenList=(32 64 128)

pMethodList=(random_init_graph_partition range_init_graph_partition) 
AggreList=(mean lstm)
walks=2
updateTimes=3

savePath=/home/cc/graph_partition_multi_layers/pseudo_mini_batch_full_batch/logs/sage/1_runs/pure_train/ogbn_products

mkdir ${savePath}

# batch_size=(70 35 18 9)
# hiddenList=(16 )
# pMethodList=(random) 
# AggreList=(mean)
layersList=(3)
fan_out_list=(25,35,40 25,35,80 25,70,80 50,70,80)
for Aggre in ${AggreList[@]}
do      
        mkdir ${savePath}/${Aggre}/
        for pMethod in ${pMethodList[@]}
        do      
                mkdir ${savePath}/${Aggre}/${pMethod}
                for layers in ${layersList[@]}
                do      
                        mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
                        for hidden in ${hiddenList[@]}
                        do
                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
                                for fan_out in ${fan_out_list[@]}
                                do
                                        nb=1
                                        for bs in ${batch_size[@]}
                                        do
                                                nb=$(($nb*2))
                                                # nb=32
                                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                # mkdir $logPath
                                                echo $logPath
                                                echo 'number of batches'
                                                echo $nb
                                                python $File \
                                                --dataset $Data \
                                                --aggre $Aggre \
                                                --seed $seed \
                                                --setseed $setseed \
                                                --GPUmem $GPUmem \
                                                --selection-method $pMethod \
                                                --batch-size $bs \
                                                --lr $lr \
                                                --num-runs $run \
                                                --num-epochs $epoch \
                                                --num-layers $layers \
                                                --num-hidden $hidden \
                                                --num-batch $nb \
                                                --dropout $dropout \
                                                --fan-out $fan_out \
                                                --walks $walks \
                                                --update-times $updateTimes \
                                                --log-indent 1 \
                                                &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
                                        done
                                done
                        done
                done
        done
done

layersList=(4)
fan_out_list=(25,35,40,40 )
for Aggre in ${AggreList[@]}
do      
        mkdir ${savePath}/${Aggre}/
        for pMethod in ${pMethodList[@]}
        do      
                mkdir ${savePath}/${Aggre}/${pMethod}
                for layers in ${layersList[@]}
                do      
                        mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
                        for hidden in ${hiddenList[@]}
                        do
                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
                                for fan_out in ${fan_out_list[@]}
                                do
                                        nb=1
                                        for bs in ${batch_size[@]}
                                        do
                                                nb=$(($nb*2))
                                                # nb=32
                                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                # mkdir $logPath
                                                echo $logPath
                                                echo 'number of batches'
                                                echo $nb
                                                python $File \
                                                --dataset $Data \
                                                --aggre $Aggre \
                                                --seed $seed \
                                                --setseed $setseed \
                                                --GPUmem $GPUmem \
                                                --selection-method $pMethod \
                                                --batch-size $bs \
                                                --lr $lr \
                                                --num-runs $run \
                                                --num-epochs $epoch \
                                                --num-layers $layers \
                                                --num-hidden $hidden \
                                                --dropout $dropout \
                                                --num-batch $nb \
                                                --fan-out $fan_out \
                                                --walks $walks \
                                                --update-times $updateTimes \
                                                --log-indent 1 \
                                                &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
                                        done
                                done
                        done
                done
        done
done

layersList=(5)
fan_out_list=(25,35,40,40,40)
for Aggre in ${AggreList[@]}
do      
        mkdir ${savePath}/${Aggre}/
        for pMethod in ${pMethodList[@]}
        do      
                mkdir ${savePath}/${Aggre}/${pMethod}
                for layers in ${layersList[@]}
                do      
                        mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
                        for hidden in ${hiddenList[@]}
                        do
                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
                                for fan_out in ${fan_out_list[@]}
                                do
                                        nb=1
                                        for bs in ${batch_size[@]}
                                        do
                                                nb=$(($nb*2))
                                                # nb=32
                                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                # mkdir $logPath
                                                echo $logPath
                                                echo 'number of batches'
                                                echo $nb
                                                python $File \
                                                --dataset $Data \
                                                --aggre $Aggre \
                                                --seed $seed \
                                                --setseed $setseed \
                                                --GPUmem $GPUmem \
                                                --selection-method $pMethod \
                                                --batch-size $bs \
                                                --lr $lr \
                                                --num-runs $run \
                                                --num-epochs $epoch \
                                                --num-layers $layers \
                                                --num-hidden $hidden \
                                                --dropout $dropout \
                                                --num-batch $nb \
                                                --fan-out $fan_out \
                                                --walks $walks \
                                                --update-times $updateTimes \
                                                --log-indent 1 \
                                                &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
                                        done
                                done
                        done
                done
        done
done

layersList=(6)
fan_out_list=(25,35,40,40,40,40)
for Aggre in ${AggreList[@]}
do      
        mkdir ${savePath}/${Aggre}/
        for pMethod in ${pMethodList[@]}
        do      
                mkdir ${savePath}/${Aggre}/${pMethod}
                for layers in ${layersList[@]}
                do      
                        mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
                        for hidden in ${hiddenList[@]}
                        do
                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
                                for fan_out in ${fan_out_list[@]}
                                do
                                        nb=1
                                        for bs in ${batch_size[@]}
                                        do
                                                nb=$(($nb*2))
                                                # nb=32
                                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                # mkdir $logPath
                                                echo $logPath
                                                echo 'number of batches'
                                                echo $nb
                                                python $File \
                                                --dataset $Data \
                                                --aggre $Aggre \
                                                --seed $seed \
                                                --setseed $setseed \
                                                --GPUmem $GPUmem \
                                                --selection-method $pMethod \
                                                --batch-size $bs \
                                                --lr $lr \
                                                --num-runs $run \
                                                --num-epochs $epoch \
                                                --num-batch $nb \
                                                --num-layers $layers \
                                                --num-hidden $hidden \
                                                --dropout $dropout \
                                                --fan-out $fan_out \
                                                --log-indent 1 \
                                                &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
                                        done
                                done
                        done
                done
        done
done
