#!/bin/bash
#SBATCH -J llama_ft
#SBATCH -N 1
#SBATCH -p savio3_gpu
#SBATCH -A co_armada2
#SBATCH -q armada2_gpu3_normal
#SBATCH --gres=gpu:A40:4
#SBATCH --time=70:00:00
#SBATCH --cpus-per-task 8
#SBATCH --output="tune-%A.out"

module load python/3.10.12-gcc-11.4.0
module load gcc/11.4.0
module load cuda/11.8.0

source /global/scratch/users/jmcavanagh/axo72324/jul/bin/activate
export WANDB_API_KEY=6a7e6c36b9bc885d48cb355afb10998284b9e8ef
export HF_HOME=/global/scratch/users/jmcavanagh/.cache/

srun accelerate launch -m axolotl.cli.train cf_lora.yml

python3 -m axolotl.cli.merge_lora /global/scratch/users/jmcavanagh/smiley/typesmiles/1b_prep/rdkit_random_smiles-1b-llama-ft/cf_lora.yml --lora_model_dir="/global/scratch/users/jmcavanagh/smiley/typesmiles/1b_prep/rdkit_random_smiles-1b-llama-ft/outputs"

