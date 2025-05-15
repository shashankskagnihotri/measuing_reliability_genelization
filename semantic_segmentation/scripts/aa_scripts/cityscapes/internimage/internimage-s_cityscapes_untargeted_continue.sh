#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=internimage-s_cityscapes_untargeted_continue
#SBATCH --output=slurm/attacks/cityscapes/upernet/internimage-s_cityscapes_untargeted_continue-%a-%A.out
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

python tools/test.py ../configs/upernet/upernet_internimage_s_512x1024_160k_cityscapes.py ../checkpoint_files/upernet/upernet_internimage_s_512x1024_160k_cityscapes.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/cityscapes/upernet/upernet_internimage_s_512x1024_160k_cityscapes/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/cityscapes/upernet/upernet_internimage_s_512x1024_160k_cityscapes/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir


end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"