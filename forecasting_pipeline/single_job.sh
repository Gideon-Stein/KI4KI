#!/bin/bash

#SBATCH --job-name=gloer_grid
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=57344
#SBATCH --cpus-per-task=24

_conda=${HOME}/anaconda3
source ${_conda}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-dam}

# works only for sbatch
# SBATCH_OPTS="${SBATCH_OPTS} --output ${RAM}"

echo $1
python /home/stein/project_repos/dam/full_baseline/forecasting_analysis/model_grid.py search=4_full_search.yaml save_path=psi_full_search  model_type=ada dir=$1 parallel=True