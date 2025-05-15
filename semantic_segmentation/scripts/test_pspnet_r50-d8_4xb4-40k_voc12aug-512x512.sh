#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=test_pspnet_r50-d8_4xb4-40k_voc12aug-512x512
#SBATCH --output=slurm/test_pspnet_r50-d8_4xb4-40k_voc12aug-512x512.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

cd mmsegmentation

python tools/test.py configs/pspnet/pspnet_r50-d8_4xb4-40k_voc12aug-512x512.py ../checkpoint_files/pspnet_r50-d8_512x512_40k_voc12aug_20200613_161222-ae9c1b8c.pth --work-dir ../work_dirs/pspnet_r50-d8_4xb4-40k_voc12aug-512x512

end='date +%s'
runtime=$((end-start))

echo Runtime: $runtime