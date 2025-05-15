#!/usr/bin/env bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=first_adversarial_attack_test
#SBATCH --output=slurm/first_adversarial_attack_test.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

cd mmsegmentation

python tools/test.py ../configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024_at.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth

end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime