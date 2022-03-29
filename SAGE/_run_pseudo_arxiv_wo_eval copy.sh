#!/bin/bash

File=pseudo_mini_batch_range_arxiv_sage.py
Data=ogbn-arxiv
epoch=30

model=sage
seed=1236 
setseed=True
GPUmem=True
lr=0.01
dropout=0.5
layers=3
hidden=256
run=1
pMethod=range
# Aggre=mean
Aggre=lstm
# fan_out_list=(10,25,15 10,25,20 10,50,100 25,35,40 50,100,200)
# fan_out_list=(10,25,10 10,25,15 10,25,20 10,50,100 25,35,40 50,100,200)
fan_out_list=(25,35,40 25,35,80 25,70,80 50,70,80)
# fan_out_list=(10,25,10)
# fan_out=25,35,50
batch_size=(45471 22736 11368 5684 2842)
# fan_out_list=(10,25,10)
# batch_size=(2842)
nb=1
# nb=$(($nb*2))
# echo $nb
# logPath=../logs/sage/1_runs/pure_train/${Aggre}/nb_${nb}/
# mkdir $logPath
for fan_out in ${fan_out_list[@]}
do
        nb=1
        for bs in ${batch_size[@]}
        do
                nb=$(($nb*2))
                # nb=32
                logPath=../logs/sage/1_runs/pure_train/${Aggre}/${pMethod}/nb_${nb}
                mkdir $logPath
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
                --fan-out $fan_out \
                &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
        done
done
