#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=unet_cityscapes_untargeted
#SBATCH --output=slurm/attacks/cityscapes/unet/unet_cityscapes_untargeted-%a-%A.out
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
        python tools/test.py ./configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py ../checkpoint_files/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/cityscapes/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/cityscapes/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
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
        python tools/test.py ./configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py ../checkpoint_files/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/cityscapes/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/cityscapes/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
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
        python tools/test.py ./configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py ../checkpoint_files/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/cityscapes/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/cityscapes/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
    done
else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"