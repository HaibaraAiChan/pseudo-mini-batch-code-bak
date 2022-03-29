#!/bin/bash
chmod u+x ../../full_batch_train/SAGE/run_full_batch_cora_wo_eval.sh
file=../..//full_batch_train/SAGE/run_full_batch_cora_wo_eval.sh
./$file
./run_pseudo_cora_wo_eval.sh