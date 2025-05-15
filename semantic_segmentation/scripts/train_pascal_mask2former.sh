#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=train_pascal_mask2former
#SBATCH --output=slurm/train_pascal_mask2former_resume_%a.out
#SBATCH --array=2-3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation

# if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
# then
#     python tools/train.py ../configs/mask2former/mask2former_swin-t_8xb2-160k_voc12aug-512x512.py --work-dir ../work_dirs/mask2former_swin-t_8xb2-160k_voc12aug-512x512
if [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    python tools/train.py ../configs/mask2former/mask2former_swin-s_8xb2-160k_voc12aug-512x512.py --work-dir ../work_dirs/mask2former_swin-s_8xb2-160k_voc12aug-512x512 --resume
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then
    python tools/train.py ../configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_voc12aug-512x512.py --work-dir ../work_dirs/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_voc12aug-512x512 --resume
else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo Runtime: $runtime