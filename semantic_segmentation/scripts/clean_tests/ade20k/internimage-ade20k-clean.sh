#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=internimage-ade20k-clean
#SBATCH --output=slurm/clean/ade20k/internimage-ade20k-clean-%a-%A.out
#SBATCH --array=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)  # Start time

cd mmsegmentation

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_t_512_160k_ade20k.py ../checkpoint_files/upernet/upernet_internimage_t_512_160k_ade20k.pth --work-dir ../clean_workdir/ade20k/upernet/upernet_internimage_t_512_160k_ade20k --show-dir ../clean_workdir/ade20k/upernet/upernet_internimage_t_512_160k_ade20k/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_s_512_160k_ade20k.py ../checkpoint_files/upernet/upernet_internimage_s_512_160k_ade20k.pth --work-dir ../clean_workdir/ade20k/upernet/upernet_internimage_s_512_160k_ade20k --show-dir ../clean_workdir/ade20k/upernet/upernet_internimage_s_512_160k_ade20k/show_dir


elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_b_512_160k_ade20k.py ../checkpoint_files/upernet/upernet_internimage_b_512_160k_ade20k.pth --work-dir ../clean_workdir/ade20k/upernet/upernet_internimage_b_512_160k_ade20k --show-dir ../clean_workdir/ade20k/upernet/upernet_internimage_b_512_160k_ade20k/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_l_640_160k_ade20k.py ../checkpoint_files/upernet/upernet_internimage_l_640_160k_ade20k.pth --work-dir ../clean_workdir/ade20k/upernet/upernet_internimage_l_640_160k_ade20k --show-dir ../clean_workdir/ade20k/upernet/upernet_internimage_l_640_160k_ade20k/show_dir

elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_xl_640_160k_ade20k.py ../checkpoint_files/upernet/upernet_internimage_xl_640_160k_ade20k.pth --work-dir ../clean_workdir/ade20k/upernet/upernet_internimage_xl_640_160k_ade20k --show-dir ../clean_workdir/ade20k/upernet/upernet_internimage_xl_640_160k_ade20k/show_dir
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]
then

    python tools/test.py ../configs/upernet/upernet_internimage_h_896_160k_ade20k.py ../checkpoint_files/upernet/upernet_internimage_h_896_160k_ade20k.pth --work-dir ../clean_workdir/ade20k/upernet/upernet_internimage_h_896_160k_ade20k --show-dir ../clean_workdir/ade20k/upernet/upernet_internimage_h_896_160k_ade20k/show_dir
else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"