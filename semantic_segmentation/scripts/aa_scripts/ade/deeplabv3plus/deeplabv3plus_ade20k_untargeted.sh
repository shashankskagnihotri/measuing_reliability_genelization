#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=deeplabv3plus_ade20k_untargeted
#SBATCH --output=slurm/deeplabv3plus_ade20k_untargeted_%a.out
#SBATCH --array=0-1
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
            python tools/test.py ./configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-160k_ade20k-512x512.py ../checkpoint_files/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_ade20k_20200615_124504-6135c7e0.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
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
            python tools/test.py ./configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py ../checkpoint_files/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
        done
    done

else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"