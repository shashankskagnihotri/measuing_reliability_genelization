#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mask2former_ade20k_clean
#SBATCH --output=slurm/clean/ade20k/mask2former_ade20k_clean_%a_%A.out
#SBATCH --array=5
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

    python tools/test.py ./configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py ../checkpoint_files/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512_20221204_000055-2d1f55f1.pth --work-dir ../clean_workdir/ade20k/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512 --show-dir ../clean_workdir/ade20k/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512/show_dir
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then

    python tools/test.py ./configs/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512.py ../checkpoint_files/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512_20221203_233905-b7135890.pth --work-dir ../clean_workdir/ade20k/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512 --show-dir ../clean_workdir/ade20k/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512/show_dir
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    
    python tools/test.py ./configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py ../checkpoint_files/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512_20221203_234230-7d64e5dd.pth --work-dir ../clean_workdir/ade20k/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512 --show-dir ../clean_workdir/ade20k/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512/show_dir
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then

    python tools/test.py ./configs/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py ../checkpoint_files/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512_20221204_143905-e715144e.pth --work-dir ../clean_workdir/ade20k/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512 --show-dir ../clean_workdir/ade20k/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512/show_dir
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]
then

    python tools/test.py ./configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640.py ../checkpoint_files/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640_20221129_125118-a4a086d2.pth --work-dir ../clean_workdir/ade20k/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640 --show-dir ../clean_workdir/ade20k/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640/show_dir
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]
then

    python tools/test.py ../configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-512x512.py /pfs/work7/workspace/scratch/ma_dschader-team_project_fss2024/benchmarking_robustness/semantic_segmentation/work_dirs/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-512x512/best_mIoU_iter_155000.pth --work-dir ../clean_workdir/ade20k/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-512x512 --show-dir ../clean_workdir/ade20k/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-512x512/show_dir


else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"