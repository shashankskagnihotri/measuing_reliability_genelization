#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=internimage_pascal_clean
#SBATCH --output=slurm/clean/pascal/upernet/internimage_pascal_clean-%a-%A.out
#SBATCH --array=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)  # Start time

cd mmsegmentation

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_s_160k_voc12aug_512x512.py ../checkpoint_files/upernet/upernet_internimage_s_160k_voc12aug_512x512.pth --work-dir ../clean_workdir/pascal/upernet/upernet_internimage_s_160k_voc12aug_512x512 --show-dir ../clean_workdir/pascal/upernet/upernet_internimage_s_160k_voc12aug_512x512/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_b_160k_voc12aug_512x512.py ../checkpoint_files/upernet/upernet_internimage_b_160k_voc12aug_512x512.pth --work-dir ../clean_workdir/pascal/upernet/upernet_internimage_b_160k_voc12aug_512x512 --show-dir ../clean_workdir/pascal/upernet/upernet_internimage_b_160k_voc12aug_512x512/show_dir
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_t_160k_voc12aug_512x512.py ../checkpoint_files/upernet/upernet_internimage_t_160k_voc12aug_512x512.pth --work-dir ../clean_workdir/pascal/upernet/upernet_internimage_t_160k_voc12aug_512x512 --show-dir ../clean_workdir/pascal/upernet/upernet_internimage_t_160k_voc12aug_512x512/show_dir

else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"