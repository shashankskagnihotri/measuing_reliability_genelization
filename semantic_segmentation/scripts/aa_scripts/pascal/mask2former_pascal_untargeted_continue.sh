#!/usr/bin/env bash
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mask2former_pascal_untargeted
#SBATCH --output=slurm/pascal/mask2former/mask2former_pascal_untargeted_continue_%a_%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)  # Start time

cd mmsegmentation

iterations=20

name="pgd"
norms=("linf" "l2")
epsilons=(8 64)
alphas=(0.01 0.1)


# Loop over norm and epsilon in parallel
for i in "${!norms[@]}"
do 
    norm="${norms[i]}"
    epsilon="${epsilons[i]}"
    alpha="${alphas[i]}"
    python tools/test.py ../configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_voc12aug-512x512.py /pfs/work7/workspace/scratch/ma_dschader-team_project_fss2024/benchmarking_robustness/semantic_segmentation/work_dirs/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_voc12aug-512x512/best_mIoU_iter_135000.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/pascal/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/pascal/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
done



end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"