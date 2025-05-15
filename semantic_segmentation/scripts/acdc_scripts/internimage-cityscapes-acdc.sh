#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=internimage-cityscapes-acdc
#SBATCH --output=slurm/acdc/cityscapes/internimage-cityscapes-acdc-%a-%A.out
#SBATCH --array=3-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)  # Start time

cd mmsegmentation

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_t_512x1024_160k_acdc.py ../checkpoint_files/upernet/upernet_internimage_t_512x1024_160k_cityscapes.pth --work-dir ../acdc_workdir/cityscapes/upernet/upernet_internimage_t_512x1024_160k_acdc --show-dir ./acdc_workdir/cityscapes/upernet/upernet_internimage_t_512x1024_160k_acdc/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_s_512x1024_160k_acdc.py ../checkpoint_files/upernet/upernet_internimage_s_512x1024_160k_cityscapes.pth --work-dir ../acdc_workdir/cityscapes/upernet/upernet_internimage_s_512x1024_160k_acdc --show-dir ./acdc_workdir/cityscapes/upernet/upernet_internimage_s_512x1024_160k_acdc/show_dir


elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_b_512x1024_160k_acdc.py ../checkpoint_files/upernet/upernet_internimage_b_512x1024_160k_cityscapes.pth --work-dir ../acdc_workdir/cityscapes/upernet/upernet_internimage_b_512x1024_160k_acdc --show-dir ./acdc_workdir/cityscapes/upernet/upernet_internimage_b_512x1024_160k_acdc/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_l_512x1024_160k_acdc.py ../checkpoint_files/upernet/upernet_internimage_l_512x1024_160k_cityscapes.pth --work-dir ../acdc_workdir/cityscapes/upernet/upernet_internimage_l_512x1024_160k_acdc --show-dir ./acdc_workdir/cityscapes/upernet/upernet_internimage_l_512x1024_160k_acdc/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_xl_512x1024_160k_acdc.py ../checkpoint_files/upernet/upernet_internimage_xl_512x1024_160k_cityscapes.pth --work-dir ../acdc_workdir/cityscapes/upernet/upernet_internimage_xl_512x1024_160k_acdc --show-dir ./acdc_workdir/cityscapes/upernet/upernet_internimage_xl_512x1024_160k_acdc/show_dir

else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"