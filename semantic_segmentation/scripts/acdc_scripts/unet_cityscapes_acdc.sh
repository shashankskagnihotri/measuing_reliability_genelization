#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4_a100,gpu_4_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=unet_cityscapes_acdc
#SBATCH --output=slurm/acdc/cityscapes/unet/unet_cityscapes_acdc-%a-%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

cd mmsegmentation 
echo "Started at $(date)";

start=$(date +%s)  # Start time

python tools/test.py ../configs/unet/unet-s5-d16_fcn_4xb4-160k_acdc-512x1024.py ../checkpoint_files/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth --work-dir ../acdc_workdir/cityscapes/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024 --show-dir ../acdc_workdir/cityscapes/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024/show_dir

end=$(date +%s)
runtime=$((end-start))

echo "Runtime: $runtime"