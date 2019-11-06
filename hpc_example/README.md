# README : Getting Started Guide to Using SLURM HPC scheduler to submit many jobs to batch scheduler

### Step 1: Understand the basic experiment: Train an autoencoder

We'd like to train an autoencoder (AE), just like in BDL class [Homework 4](https://www.cs.tufts.edu/comp/150BDL/2018f/assignments/hw4.html)

We're interested in exploring several settings:
* learning rate (lr) of 0.010 and 0.001
* number of hidden units of 032, 128, and 512

Suppose we've got a script that can train an AE under different settings: [`hw4_ae.py`](https://github.com/tufts-ml/comp150_bdl_2018f_public/blob/master/hpc_example/hw4_ae.py)

Recall the Usage of this script:
```
$ python hw4_ae.py --help
  --help            show this help message and exit
  --n_epochs N_EPOCHS   number of epochs (default: 2)
  --batch_size BATCH_SIZE
                        batch size (default: 100)
  --lr LR               Learning rate for grad. descent (default: 0.001)
  --hidden_layer_sizes HIDDEN_LAYER_SIZES
                        Comma-separated list of size values (default: "32")
  --filename_prefix FILENAME_PREFIX
  --q_sigma Q_SIGMA     Fixed variance of approximate posterior (default: 0.1)
  --n_mc_samples N_MC_SAMPLES
                        Number of Monte Carlo samples (default: 1)
  --seed SEED           random seed (default: 8675309)

```

So we could simply just manually call this script at different settings, like
```
python hw4_ae.py \
    --lr 0.001 \
    --hidden_layer_sizes 32 \
    --filename_prefix myresult
```

But this is boring! Let's use the cluster to run all 6 jobs (2 lr settings, 3 arch size settings) simultaneously!

### Step 2: Create a "do_experiment.slurm" script to perform our work

Take a look at [do_experiment.slurm](https://github.com/tufts-ml/comp150_bdl_2018f_public/blob/master/hpc_example/do_experiment.slurm)

You'll see it's like a standard shell script, but with a weird header (lines that start with '#')

The main body should look familiar:
* load the conda environment
* call the python script
* clean up after itself (deactivate the conda environment)

We can ignore the header for now. Try it out! It's just like any shell script:

```
$ bash do_experiment.slurm lr=0.001 hidden_layer_sizes=032
```

EXPECTED OUT:

```
Saving with prefix: mydemo
==== evaluation after epoch 0
Total images 60000. Total on pixels: 6221431. Frac pixels on: 0.132
  epoch   0  train loss 0.701  bce 0.701  l1 0.502
Total images 10000. Total on pixels: 1052359. Frac pixels on: 0.134
  epoch   0  test  loss 0.701  bce 0.701  l1 0.502
====  done with eval at epoch 0
  epoch   1 | frac_seen 0.100 | avg loss 4.824e-03 | batch loss  2.844e-03 | batch l1  0.185
  epoch   1 | frac_seen 0.200 | avg loss 3.768e-03 | batch loss  2.659e-03 | batch l1  0.175
...
```

Great! But what's the big deal? This is a wrapper for our hw4_ae.py script that *can be understood* by the SLURM job scheduling system.

Look at the header:
```
#!/usr/bin/env bash
#SBATCH -n 1                # Number of cores
#SBATCH -t 0-05:00          # Runtime in D-HH:MM
#SBATCH -p batch            # Partition to submit to
#SBATCH --mem-per-cpu 2000  # Memory (in MB) per cpu
#SBATCH -o log_%j.out       # Write stdout to file named log_JOBID.out in current dir
#SBATCH -e log_%j.err       # Write stderr to file named log_JOBID.err in current dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

```
All this says that that when we request a job, we want 1 core, to run for at most 5 hours, and use at most 2GB (2000 MB) of RAM.


### Step 3: Try out a single job submission via 'sbatch'

Once we have our do_experiment.slurm script, we can **submit** it to the job scheduler via this command:

```
$ lr=0.01 hidden_layer_sizes=32 sbatch < do_experiment.slurm
```
Note: the env var arguments need to go FIRST when calling sbatch.

EXPECTED OUTPUT:
```
Submitted batch job 34740124
```
Remember that number (in this case 34740124). This is the JOBID. 

OK, the job has been submitted. You can check on it with the command:
```
$ squeue -u TUFTS_USERNAME
```
EXPECTED OUTPUT:
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          34740124     batch   sbatch mhughe02  R       1:24      1 m3n46
          34740078 interacti     bash mhughe02  R    1:10:58      1 alpha001
```
If you see status of 'R' and a nodelist that looks like 'm3n46', then congrats! Your job is running!

You might alternatively see a status like 'PENDING' if the queue is very busy. Be patient!

#### How can I monitor my job while it runs?

You can use the `squeue -u TUFTS_USERNAME` call to check on all your jobs. When they are done, they will no longer appear in that list.

You can also look at the log_JOBID.out and log_JOBID.err log files, which are capturing the stdout and stderr of your jobs!

```
$ cat log_34740124.out
Saving with prefix: mydemo
==== evaluation after epoch 0
Total images 60000. Total on pixels: 6221431. Frac pixels on: 0.132
  epoch   0  train loss 0.701  bce 0.701  l1 0.502
Total images 10000. Total on pixels: 1052359. Frac pixels on: 0.134
  epoch   0  test  loss 0.701  bce 0.701  l1 0.502
====  done with eval at epoch 0
  epoch   1 | frac_seen 0.100 | avg loss 2.989e-03 | batch loss  2.405e-03 | batch l1  0.159
  epoch   1 | frac_seen 0.200 | avg loss 2.672e-03 | batch loss  2.254e-03 | batch l1  0.147
...
```

### Step 3: How to launch many jobs at once

We'll need two scripts:
* one to loop over all settings [launch_experiments.sh](https://github.com/tufts-ml/comp150_bdl_2018f_public/blob/master/hpc_example/launch_experiments.sh)
* one to do the work at each setting [do_experiment.slurm](https://github.com/tufts-ml/comp150_bdl_2018f_public/blob/master/hpc_example/do_experiment.slurm)


Our desired end behavior is to just call the "launch_experiments.sh" script with a desired action:
```
$ bash launch_experiments.sh list      ## Just list out the settings we'll explore
$ bash launch_experiments.sh run_here  ## Run each setting one-at-a-time here in this terminal (useful for debugging)
$ bash launch_experiments.sh submit    ## Send the work to the HPC cluster to be scheduled, via 'sbatch'
```

As a test, please try the first command at your terminal. Don't try the others just yet.

```
$ bash launch_experiments.sh
lr=0.001  hidden_layer_sizes=032
lr=0.001  hidden_layer_sizes=128
lr=0.001  hidden_layer_sizes=512
lr=0.010  hidden_layer_sizes=032
lr=0.010  hidden_layer_sizes=128
lr=0.010  hidden_layer_sizes=512
```

Great! It's listing out all the settings we want to experiment with.

If you peek at launch_experiment.sh, you'll see that we:
* loop over all settings of the variables
* at each one call do_experiment.sh 

Note that we are using [Environment Variables](https://www.digitalocean.com/community/tutorials/how-to-read-and-set-environmental-and-shell-variables-on-a-linux-vps) to store and pass information between the two scripts.

There's a simple IF statement that controls whether we call bash and run locally (action='run_here') or call sbatch and let the grid do the work (action='submit').

OK, so let's try it! 

```
$ bash launch_experiments.sh submit
lr=0.001  hidden_layer_sizes=032
Submitted batch job 34740131
lr=0.001  hidden_layer_sizes=128
Submitted batch job 34740132
lr=0.001  hidden_layer_sizes=512
Submitted batch job 34740133
lr=0.010  hidden_layer_sizes=032
Submitted batch job 34740134
lr=0.010  hidden_layer_sizes=128
Submitted batch job 34740135
lr=0.010  hidden_layer_sizes=512
Submitted batch job 34740136
```

Tada! You've submitted your first set of batch jobs!
