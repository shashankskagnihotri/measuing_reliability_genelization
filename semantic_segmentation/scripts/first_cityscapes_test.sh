#!/usr/bin/env bash
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=first_cityscapes_test
#SBATCH --output=slurm/first_cityscapes_test.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

source activate semseg
cd mmsegmentation

python tools/test.py configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py ../checkpoint_files/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth --work-dir ../work_dirs/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024

end='date +%s'
runtime=$((end-start))

echo Runtime: $runtime