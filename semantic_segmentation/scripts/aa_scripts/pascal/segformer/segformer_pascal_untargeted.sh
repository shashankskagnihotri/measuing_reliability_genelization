#!/usr/bin/env bash
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=segformer_ade20k_untargeted
#SBATCH --output=slurm/attacks/pascal/segformer_MIT-B%a_ade20k_untargeted_%A.out
#SBATCH --array=3-5
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
    
    # Loop over name
    for name in "${names[@]}"
    do
        # Loop over norm and epsilon in parallel
        for i in "${!norms[@]}"
        do 
            norm="${norms[i]}"
            epsilon="${epsilons[i]}"
            alpha="${alphas[i]}"
            python tools/test.py ../configs/segformer/segformer_mit-b0_8xb2-160k_voc12aug-512x512.py ../work_dirs/segformer_mit-b0_8xb2-160k_voc12aug-512x512/iter_160000.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/pascal/segformer/segformer_mit-b0_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/pascal/segformer/segformer_mit-b0_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
        done
    done
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then
    
    # Loop over name
    for name in "${names[@]}"
    do
        # Loop over norm and epsilon in parallel
        for i in "${!norms[@]}"
        do 
            norm="${norms[i]}"
            epsilon="${epsilons[i]}"
            alpha="${alphas[i]}"
            python tools/test.py ../configs/segformer/segformer_mit-b1_8xb2-160k_voc12aug-512x512.py ../work_dirs/segformer_mit-b1_8xb2-160k_voc12aug-512x512/iter_160000.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/pascal/segformer/segformer_mit-b1_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/pascal/segformer/segformer_mit-b1_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
        done
    done
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    
    # Loop over name
    for name in "${names[@]}"
    do
        # Loop over norm and epsilon in parallel
        for i in "${!norms[@]}"
        do 
            norm="${norms[i]}"
            epsilon="${epsilons[i]}"
            alpha="${alphas[i]}"
            python tools/test.py ../configs/segformer/segformer_mit-b2_8xb2-160k_voc12aug-512x512.py ../work_dirs/segformer_mit-b2_8xb2-160k_voc12aug-512x512/iter_160000.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/pascal/segformer/segformer_mit-b2_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/pascal/segformer/segformer_mit-b2_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
        done
    done
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then
    
    # Loop over name
    for name in "${names[@]}"
    do
        # Loop over norm and epsilon in parallel
        for i in "${!norms[@]}"
        do
            norm="${norms[i]}"
            epsilon="${epsilons[i]}"
            alpha="${alphas[i]}"
            python tools/test.py ../configs/segformer/segformer_mit-b3_8xb2-160k_voc12aug-512x512.py ../work_dirs/segformer_mit-b3_8xb2-160k_voc12aug-512x512/iter_160000.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/pascal/segformer/segformer_mit-b3_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/pascal/segformer/segformer_mit-b3_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
        done
    done
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]
then
    
    # Loop over name
    for name in "${names[@]}"
    do
        # Loop over norm and epsilon in parallel
        for i in "${!norms[@]}"
        do 
            norm="${norms[i]}"
            epsilon="${epsilons[i]}"
            alpha="${alphas[i]}"
            python tools/test.py ../configs/segformer/segformer_mit-b4_8xb2-160k_voc12aug-512x512.py ../work_dirs/segformer_mit-b4_8xb2-160k_voc12aug-512x512/iter_160000.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/pascal/segformer/segformer_mit-b4_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/pascal/segformer/segformer_mit-b4_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
        done
    done
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]
then
    
    # Loop over name
    for name in "${names[@]}"
    do
        # Loop over norm and epsilon in parallel
        for i in "${!norms[@]}"
        do 
            norm="${norms[i]}"
            epsilon="${epsilons[i]}"
            alpha="${alphas[i]}"
            python tools/test.py ../configs/segformer/segformer_mit-b5_8xb2-160k_voc12aug-512x512.py ../work_dirs/segformer_mit-b5_8xb2-160k_voc12aug-512x512/iter_160000.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/pascal/segformer/segformer_mit-b5_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/pascal/segformer/segformer_mit-b5_8xb2-160k_voc12aug-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
        done
    done
else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"