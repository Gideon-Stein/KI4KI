#!/usr/bin/env bash



#item_list=('ASCE_015_06' 'ASCE_088_03' 'VERTICAL' 'DESC_139_08' 'EASTWEST' 'DESC_037_04' 'DESC_139_07' 'ASCE_088_02')



# values for a single GPU
N_GPUS=0
CPUS=8
RAM=8

SBATCH_OPTS="${SBATCH_OPTS} --gres gpu:${N_GPUS}"
SBATCH_OPTS="${SBATCH_OPTS} -c ${CPUS}"
SBATCH_OPTS="${SBATCH_OPTS} --mem ${RAM}G"
SBATCH_OPTS="${SBATCH_OPTS} --partition workstation,robolab"



for i in "linear" "ada" "forest" "sarimax"
    do
        echo $i
        srun ${SBATCH_OPTS} "python /home/stein/project_repos/dam/full_baseline/forecasting_analysis/model_grid.py search=2_added_fourier.yaml save_path=results/added_fourier model_type=$i  dir=lot"
    done