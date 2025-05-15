#!/usr/bin/env bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=all_untargeted_attacks_cityscapes_segformer_until_array_100.sh
#SBATCH --output=slurm/all_untargeted_attacks_cityscapes_segformer_until_array_100_%a.out
#SBATCH --array=1-100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 7 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 8 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 9 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 10 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 11 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 12 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 13 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 14 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 15 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 16 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 17 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 18 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 19 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 20 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 21 ]]
then
    python tools/test.py ../segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024/iter_144000.pth --work-dir ../work_dirs/segformer_mit-b0_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 22 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 23 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 24 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 25 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 26 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 27 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 28 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 29 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 30 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 31 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 32 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 33 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 34 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 35 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 36 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 37 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 38 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 39 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 40 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 41 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 42 ]]
then
    python tools/test.py ../segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b1_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 43 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 44 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 45 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 46 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 47 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 48 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 49 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 50 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 51 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 52 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 53 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 54 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 55 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 56 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 57 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 58 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 59 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 60 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 61 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 62 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 63 ]]
then
    python tools/test.py ../segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b2_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 64 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 65 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 66 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 67 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 68 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 69 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 70 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 71 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 72 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 73 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 74 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 75 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 76 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 77 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 78 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 79 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 80 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 81 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 82 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 83 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 84 ]]
then
    python tools/test.py ../segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b3_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 85 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 86 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 87 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 88 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 89 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 90 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 91 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 92 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 93 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 94 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 95 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 96 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 97 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 98 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 99 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 100 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo Runtime: $runtime