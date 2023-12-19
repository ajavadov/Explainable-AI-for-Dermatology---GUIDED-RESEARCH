#!/bin/bash -l

#SBATCH --job-name=dermaproj1
#SBATCH --output=res.txt
#SBATCH --error=res.err

#SBATCH --cpus-per-task=8
#SBATCH --gpus=1


source activate dermaproj1
srun ./run.sh
