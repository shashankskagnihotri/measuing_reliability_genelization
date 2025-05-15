#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mask2former_cityscapes_clean
#SBATCH --output=slurm/clean/cityscapes/mask2former_cityscapes_clean_%a_%A.out
#SBATCH --array=3
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

    python tools/test.py ./configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py ../checkpoint_files/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024_20221202_140802-ffd9d750.pth --work-dir ../clean_workdir/cityscapes/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024 --show-dir ../clean_workdir/cityscapes/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024/show_dir
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then

    python tools/test.py ./configs/mask2former/mask2former_r101_8xb2-90k_cityscapes-512x1024.py ../checkpoint_files/mask2former/mask2former_r101_8xb2-90k_cityscapes-512x1024_20221130_031628-43e68666.pth --work-dir ../clean_workdir/cityscapes/mask2former/mask2former_r101_8xb2-90k_cityscapes-512x1024 --show-dir ../clean_workdir/cityscapes/mask2former/mask2former_r101_8xb2-90k_cityscapes-512x1024/show_dir
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    
    python tools/test.py ./configs/mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024.py ../checkpoint_files/mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024_20221127_144501-36c59341.pth --work-dir ../clean_workdir/cityscapes/mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024 --show-dir ../clean_workdir/cityscapes/mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024/show_dir
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then

    python tools/test.py ./configs/mask2former/mask2former_swin-s_8xb2-90k_cityscapes-512x1024.py ../checkpoint_files/mask2former/mask2former_swin-s_8xb2-90k_cityscapes-512x1024_20221127_143802-9ab177f6.pth --work-dir ../clean_workdir/cityscapes/mask2former/mask2former_swin-s_8xb2-90k_cityscapes-512x1024 --show-dir ../clean_workdir/cityscapes/mask2former/mask2former_swin-s_8xb2-90k_cityscapes-512x1024/show_dir


else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"