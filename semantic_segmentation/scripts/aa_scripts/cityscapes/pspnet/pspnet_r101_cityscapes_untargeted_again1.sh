#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pspnet_r101_cityscapes_untargeted_again1
#SBATCH --output=slurm/attakcs/cityscapes/pspnet_r101_cityscapes_untargeted_again1_%a_%A.out
#SBATCH --array=0-2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)  # Start time

cd mmsegmentation

iterations=20

names=("pgd" "cospgd" "segpgd")
norms=("linf" "l2")
epsilons=(8 64)
alphas=(0.01 0.1)

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then
    name="pgd"
    # Loop over norm and epsilon in parallel
    for i in "${!norms[@]}"
    do 
        norm="${norms[i]}"
        epsilon="${epsilons[i]}"
        alpha="${alphas[i]}"
        python tools/test.py ./configs/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024.py ../checkpoint_files/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
    done
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then
    name="cospgd"
    # Loop over norm and epsilon in parallel
    for i in "${!norms[@]}"
    do 
        norm="${norms[i]}"
        epsilon="${epsilons[i]}"
        alpha="${alphas[i]}"
        python tools/test.py ./configs/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024.py ../checkpoint_files/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
    done
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    name="segpgd"
    # Loop over norm and epsilon in parallel
    for i in "${!norms[@]}"
    do 
        norm="${norms[i]}"
        epsilon="${epsilons[i]}"
        alpha="${alphas[i]}"
        python tools/test.py ./configs/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024.py ../checkpoint_files/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
    done
else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"