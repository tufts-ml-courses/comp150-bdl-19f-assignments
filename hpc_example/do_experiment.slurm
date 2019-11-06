#!/usr/bin/env bash
#SBATCH -n 1                # Number of cores
#SBATCH -t 0-05:00          # Runtime in D-HH:MM
#SBATCH -p batch            # Partition to submit to
#SBATCH --mem-per-cpu 2000  # Memory (in MB) per cpu
#SBATCH -o log_%j.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e log_%j.err       # Write stderr to file named log_JOBIDNUM.err in current dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

# Load the right conda environment
source activate bdl_pytorch_readonly

# If some ENV variables are not defined, set them to defaults
if [[ -z $lr ]]; then
    lr=0.001;
fi
if [[ -z $hidden_layer_sizes ]]; then
    hidden_layer_sizes=32;
fi
if [[ -z $n_epochs ]]; then
    n_epochs=2;
fi
if [[ -z $filename_prefix ]]; then
    filename_prefix='mydemo'
fi

# Pass along all ENV variables as arguments to my Python script
python hw4_ae.py \
    --lr $lr \
    --hidden_layer_sizes $hidden_layer_sizes \
    --filename_prefix $filename_prefix \
    --n_epochs $n_epochs \

source deactivate


