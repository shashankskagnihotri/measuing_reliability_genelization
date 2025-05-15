#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=internimage-h_ade20k_untargeted_continue
#SBATCH --output=slurm/attacks/ade/upernet/internimage-h_ade20k_untargeted-%a-%A.out
#SBATCH --array=0-2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)  # Start time

cd mmsegmentation

iterations=20

names=("pgd" "cospgd" "segpgd")
norm="l2"
epsilon=64
alpha=0.1


if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then

    name="pgd"

    python tools/test.py ../configs/upernet/upernet_internimage_h_896_160k_ade20k.py ../checkpoint_files/upernet/upernet_internimage_h_896_160k_ade20k.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/ade/upernet/upernet_internimage_h_896_160k_ade20k/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/ade/upernet/upernet_internimage_h_896_160k_ade20k/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then

    name="cospgd"
    
    python tools/test.py ../configs/upernet/upernet_internimage_h_896_160k_ade20k.py ../checkpoint_files/upernet/upernet_internimage_h_896_160k_ade20k.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/ade/upernet/upernet_internimage_h_896_160k_ade20k/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/ade/upernet/upernet_internimage_h_896_160k_ade20k/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then

    name="segpgd"
    

    python tools/test.py ../configs/upernet/upernet_internimage_h_896_160k_ade20k.py ../checkpoint_files/upernet/upernet_internimage_h_896_160k_ade20k.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/ade/upernet/upernet_internimage_h_896_160k_ade20k/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --show-dir ../aa_workdir/ade/upernet/upernet_internimage_h_896_160k_ade20k/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"