#!/bin/bash
#SBATCH --job-name=gdro_models
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-11
#SBATCH --output=logs/mortality_miivtozigong/dro_model_%A_%a.out
#SBATCH --error=logs/mortality_miivtozigong/dro_model_%A_%a.err

# show job information
echo "Running in: $(pwd)"

# copy dataset into local scratch: 
# IMPORTANT: change username!
cp /cluster/work/math/username/data/miiv.parquet $TMPDIR/
cp /cluster/work/math/username/data/zigong.parquet $TMPDIR/

# activate Python environment
# IMPORTANT: change username! (or change this to your conda installation path as needed)
source /cluster/home/username/nobackup/miniforge3/etc/profile.d/conda.sh
conda activate cvx-env

# run model
python mortality_miivtozigong.py