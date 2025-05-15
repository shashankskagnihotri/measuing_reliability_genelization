#!/usr/bin/env bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=deeplabv3_r101_pascal_untargeted_again
#SBATCH --output=slurm/attacks/pascal/deeplabv3_r101_pascal_untargeted_again_%a_%A.out
#SBATCH --array=0-1
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


python tools/test.py ./configs/deeplabv3/deeplabv3_r101-d8_4xb4-40k_voc12aug-512x512.py ../checkpoint_files/deeplabv3/deeplabv3_r101-d8_512x512_40k_voc12aug_20200613_161432-0017d784.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/pascal/deeplabv3/deeplabv3_r101-d8_4xb4-40k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/pascal/deeplabv3/deeplabv3_r101-d8_4xb4-40k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"