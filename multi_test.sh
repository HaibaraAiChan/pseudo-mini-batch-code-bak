#!/bin/bash


epoch=30
Aggre=mean
model=sage
run=1

File=pseudo_mini_batch_range_reddit_sage.py
Data=reddit
# batch_size=( 76716 38357  19178 9590 4795 2400 1200)
batch_size=(76716)
layers=2
weightdecay=5e-4
lr=0.01
dropout=0.5
fan_out=25,35
seed=1236 
hidden=16
# # with mem info in 1 run logs

for bs in ${batch_size[@]}
do
        python $File \
        --dataset $Data \
        --aggre $Aggre \
        --seed $seed \
        --selection-method range \
        --batch-size $bs \
        --weight-decay $weightdecay \
        --lr $lr \
        --num-runs $run \
        --num-epochs $epoch \
        --num-layers $layers \
        --num-hidden $hidden \
        --dropout $dropout \
        --fan-out $fan_out \
        &> logs/sage/1_runs/${Data}_${Aggre}_${seed}_layers_${layers}_nb_2_run_${run}.log
done

File=pseudo_mini_batch_range_citation_sage.py
Data=pubmed
batch_size=(30)
layers=2
dropout=0.5
fan_out=25,35
hidden=16
seed=1238
# # with mem info in 1 run logs

for bs in ${batch_size[@]}
do
        python $File \
        --dataset $Data \
        --aggre $Aggre \
        --seed $seed \
        --selection-method range \
        --batch-size $bs \
        --weight-decay $weightdecay \
        --lr $lr \
        --num-runs $run \
        --num-epochs $epoch \
        --num-layers $layers \
        --num-hidden $hidden \
        --dropout $dropout \
        --fan-out $fan_out \
        &> logs/sage/1_runs/${Data}_${Aggre}_${seed}_layers_${layers}_nb_2_run_${run}.log
done

Data=cora
batch_size=(70)
seed=1236
for bs in ${batch_size[@]}
do
        python $File \
        --dataset $Data \
        --aggre $Aggre \
        --seed $seed \
        --selection-method range \
        --batch-size $bs \
        --weight-decay $weightdecay \
        --lr $lr \
        --num-runs $run \
        --num-epochs $epoch \
        --num-layers $layers \
        --num-hidden $hidden \
        --dropout $dropout \
        --fan-out $fan_out \
        &> logs/sage/1_runs/${Data}_${Aggre}_${seed}_layers_${layers}_nb_2_run_${run}.log
done

File=pseudo_mini_batch_range_arxiv_sage.py
Data=ogbn-arxiv
batch_size=(45471)
layers=3
weightdecay=5e-4
lr=0.01
dropout=0.5
fan_out=25,35,40
seed=1236
hidden=256
# with mem info in 1 run logs

for bs in ${batch_size[@]}
do
        python $File \
        --dataset $Data \
        --aggre $Aggre \
        --seed $seed \
        --selection-method range \
        --batch-size $bs \
        --weight-decay $weightdecay \
        --lr $lr \
        --num-runs $run \
        --num-epochs $epoch \
        --num-layers $layers \
        --num-hidden $hidden \
        --dropout $dropout \
        --fan-out $fan_out &> logs/sage/1_runs/${Data}_${Aggre}_${seed}_layers_${layers}_nb_2_run_${run}.log
done
File=pseudo_mini_batch_range_products_sage.py
Data=ogbn-products
batch_size=(98308)
hidden=64
for bs in ${batch_size[@]}
do
        python $File \
        --dataset $Data \
        --aggre $Aggre \
        --seed $seed \
        --selection-method range \
        --batch-size $bs \
        --weight-decay $weightdecay \
        --lr $lr \
        --num-runs $run \
        --num-epochs $epoch \
        --num-layers $layers \
        --num-hidden $hidden \
        --dropout $dropout \
        --fan-out $fan_out &> logs/sage/1_runs/${Data}_${Aggre}_${seed}_layers_${layers}_nb_2_run_${run}.log
done
