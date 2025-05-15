#!/usr/bin/env bash
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mask2former_pascal_clean
#SBATCH --output=slurm/clean/pascal/mask2former/mask2former_pascal_clean_%a_%A.out
#SBATCH --array=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)  # Start time

cd mmsegmentation

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then

    python tools/test.py ../configs/mask2former/mask2former_swin-t_8xb2-160k_voc12aug-512x512.py /pfs/work7/workspace/scratch/ma_dschader-team_project_fss2024/benchmarking_robustness/semantic_segmentation/work_dirs/mask2former_swin-t_8xb2-160k_voc12aug-512x512/best_mIoU_iter_140000.pth --work-dir ../clean_workdir/pascal/mask2former/mask2former_swin-t_8xb2-160k_voc12aug-512x512 --show-dir ../clean_workdir/pascal/mask2former/mask2former_swin-t_8xb2-160k_voc12aug-512x512/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then

    python tools/test.py ../configs/mask2former/mask2former_swin-s_8xb2-160k_voc12aug-512x512.py /pfs/work7/workspace/scratch/ma_dschader-team_project_fss2024/benchmarking_robustness/semantic_segmentation/work_dirs/mask2former_swin-s_8xb2-160k_voc12aug-512x512/best_mIoU_iter_155000.pth --work-dir ../clean_workdir/pascal/mask2former/mask2former_swin-s_8xb2-160k_voc12aug-512x512 --show-dir ../clean_workdir/pascal/mask2former/mask2former_swin-s_8xb2-160k_voc12aug-512x512/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then

    python tools/test.py ../configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_voc12aug-512x512.py /pfs/work7/workspace/scratch/ma_dschader-team_project_fss2024/benchmarking_robustness/semantic_segmentation/work_dirs/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_voc12aug-512x512/best_mIoU_iter_135000.pth --work-dir ../clean_workdir/pascal/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_voc12aug-512x512 --show-dir ../clean_workdir/pascal/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_voc12aug-512x512/show_dir

else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"