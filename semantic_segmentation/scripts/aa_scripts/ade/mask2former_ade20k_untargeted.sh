#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mask2former_ade20k_untargeted
#SBATCH --output=slurm/ade/mask2former/mask2former_ade20k_untargeted_MIT-B%a.out
#SBATCH --array=0-5
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
            python tools/test.py ./configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py ../checkpoint_files/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512_20221204_000055-2d1f55f1.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
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
            python tools/test.py ./configs/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512.py ../checkpoint_files/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512_20221203_233905-b7135890.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
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
            python tools/test.py ./configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py ../checkpoint_files/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512_20221203_234230-7d64e5dd.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
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
            python tools/test.py ./configs/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py ../checkpoint_files/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512_20221204_143905-e715144e.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
        
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
            python tools/test.py ./configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640.py ../checkpoint_files/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640_20221129_125118-a4a086d2.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
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
            python tools/test.py ../configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-512x512.py ../work_dirs/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-512x512/best_mIoU_iter_155000.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
        done
    done
else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"