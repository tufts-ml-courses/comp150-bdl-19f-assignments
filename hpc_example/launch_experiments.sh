#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export n_epochs=3
export batch_size=100
export n_mc_samples=1
export q_sigma=0.1

## Learning rate for adam
for lr in 0.001 0.010
do
    export lr=$lr

## Architecture size (num hidden units)
for arch in 032 128 512
do
    export hidden_layer_sizes=$arch
    export filename_prefix="mydemo-lr=$lr-arch=$arch"

    ## Use this line to see where you are in the loop
    echo "lr=$lr  hidden_layer_sizes=$hidden_layer_sizes"

    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < do_experiment.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash do_experiment.slurm
    fi

done
done


