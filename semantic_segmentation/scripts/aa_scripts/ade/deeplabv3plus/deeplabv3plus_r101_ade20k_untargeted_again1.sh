#!/usr/bin/env bash
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=deeplabv3plus_r101_ade20k_untargeted_again1
#SBATCH --output=slurm/deeplabv3plus_r101_ade20k_untargeted_again1_%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)  # Start time

cd mmsegmentation

iterations=20
name="segpgd"
norm="l2"
epsilon=64
alpha=0.1

python tools/test.py ./configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py ../checkpoint_files/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"