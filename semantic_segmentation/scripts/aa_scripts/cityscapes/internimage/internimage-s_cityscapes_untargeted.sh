#!/usr/bin/env bash
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=internimage-s_cityscapes_untargeted
#SBATCH --output=slurm/attacks/cityscapes/upernet/internimage-s_cityscapes_untargeted-%a-%A.out
#SBATCH --array=1-2
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
        python tools/test.py ../configs/upernet/upernet_internimage_s_512x1024_160k_cityscapes.py ../checkpoint_files/upernet/upernet_internimage_s_512x1024_160k_cityscapes.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/cityscapes/upernet/upernet_internimage_s_512x1024_160k_cityscapes/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/cityscapes/upernet/upernet_internimage_s_512x1024_160k_cityscapes/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
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
        python tools/test.py ../configs/upernet/upernet_internimage_s_512x1024_160k_cityscapes.py ../checkpoint_files/upernet/upernet_internimage_s_512x1024_160k_cityscapes.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/cityscapes/upernet/upernet_internimage_s_512x1024_160k_cityscapes/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/cityscapes/upernet/upernet_internimage_s_512x1024_160k_cityscapes/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
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
        python tools/test.py ../configs/upernet/upernet_internimage_s_512x1024_160k_cityscapes.py ../checkpoint_files/upernet/upernet_internimage_s_512x1024_160k_cityscapes.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/cityscapes/upernet/upernet_internimage_s_512x1024_160k_cityscapes/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/cityscapes/upernet/upernet_internimage_s_512x1024_160k_cityscapes/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
    done
else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"