#!/usr/bin/env bash
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mask2former_cityscapes_untargeted
#SBATCH --output=slurm/cityscapes/mask2former/mask2former_cityscapes_untargeted_MIT-B%a.out
#SBATCH --array=1-3
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

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then
    # Loop over norm and epsilon in parallel
    for i in "${!norms[@]}"
    do 
        norm="${norms[i]}"
        epsilon="${epsilons[i]}"
        alpha="${alphas[i]}"
        python tools/test.py ./configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py ../checkpoint_files/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024_20221202_140802-ffd9d750.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/cityscapes/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/cityscapes/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
    done
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then
    # Loop over norm and epsilon in parallel
    for i in "${!norms[@]}"
    do 
        norm="${norms[i]}"
        epsilon="${epsilons[i]}"
        alpha="${alphas[i]}"
        python tools/test.py ./configs/mask2former/mask2former_r101_8xb2-90k_cityscapes-512x1024.py ../checkpoint_files/mask2former/mask2former_r101_8xb2-90k_cityscapes-512x1024_20221130_031628-43e68666.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/cityscapes/mask2former/mask2former_r101_8xb2-90k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/cityscapes/mask2former/mask2former_r101_8xb2-90k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
    done
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    
    # Loop over norm and epsilon in parallel
    for i in "${!norms[@]}"
    do 
        norm="${norms[i]}"
        epsilon="${epsilons[i]}"
        alpha="${alphas[i]}"
        python tools/test.py ./configs/mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024.py ../checkpoint_files/mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024_20221127_144501-36c59341.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/cityscapes/mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/cityscapes/mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
    done
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then

    # Loop over norm and epsilon in parallel
    for i in "${!norms[@]}"
    do 
        norm="${norms[i]}"
        epsilon="${epsilons[i]}"
        alpha="${alphas[i]}"
        python tools/test.py ./configs/mask2former/mask2former_swin-s_8xb2-90k_cityscapes-512x1024.py ../checkpoint_files/mask2former/mask2former_swin-s_8xb2-90k_cityscapes-512x1024_20221127_143802-9ab177f6.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/cityscapes/mask2former/mask2former_swin-s_8xb2-90k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/cityscapes/mask2former/mask2former_swin-s_8xb2-90k_cityscapes-512x1024/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
    
    done

else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"