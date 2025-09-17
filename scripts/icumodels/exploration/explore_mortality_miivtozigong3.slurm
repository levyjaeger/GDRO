#!/bin/bash
#SBATCH --job-name=gdro_models
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-5
#SBATCH --output=logs/explorations/dro_model_%A_%a.out
#SBATCH --error=logs/explorations/dro_model_%A_%a.err

# show job information
echo "Running in: $(pwd)"

# copy dataset into local scratch
cp /cluster/work/math/jaegerl/02_data/02_data_input/miiv.parquet $TMPDIR/
cp /cluster/work/math/jaegerl/02_data/02_data_input/zigong.parquet $TMPDIR/

# activate environment
source /cluster/home/jaegerl/nobackup/miniforge3/etc/profile.d/conda.sh
conda activate cvx-env

# run model
python explore_mortality_miivtozigong3.py