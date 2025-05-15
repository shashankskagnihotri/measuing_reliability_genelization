#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mask2former_ade20k_untargeted_additional
#SBATCH --output=slurm/ade/mask2former/mask2former_ade20k_untargeted_additonal.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)  # Start time

cd mmsegmentation

name="pgd"
norm="linf"
epsilon=2
iterations=40
alpha=0.01
python tools/test.py ./configs/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512.py ../checkpoint_files/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512_20221203_233905-b7135890.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"