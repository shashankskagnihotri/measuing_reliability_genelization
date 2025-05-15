#!/usr/bin/env bash
#SBATCH --time=25:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=train_mask2former_swin-b-in1k-384x384-pre_8xb2-90k_cityscapes-512x1024
#SBATCH --output=slurm/train_mask2former_swin-b-in1k-384x384-pre_8xb2-90k_cityscapes-512x1024.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

cd mmsegmentation

python tools/train.py ./configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-90k_cityscapes-512x1024.py --work-dir ../work_dirs/mask2former_swin-b-in1k-384x384-pre_8xb2-90k_cityscapes-512x1024

end='date +%s'
runtime=$((end-start))

echo Runtime: $runtime