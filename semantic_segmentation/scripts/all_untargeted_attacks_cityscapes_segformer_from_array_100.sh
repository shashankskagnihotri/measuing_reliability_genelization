#!/usr/bin/env bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=all_untargeted_attacks_cityscapes_segformer_from_array_100.sh
#SBATCH --output=slurm/all_untargeted_attacks_cityscapes_segformer_from_array_100_%a.out
#SBATCH --array=1-26
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]
then
    python tools/test.py ../segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b4_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 7 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 8 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 9 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 10 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 11 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 12 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='pgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 13 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 14 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 15 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 16 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 17 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 18 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 19 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='cospgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 20 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 21 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 22 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 23 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=5
elif [[ $SLURM_ARRAY_TASK_ID -eq 24 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=10
elif [[ $SLURM_ARRAY_TASK_ID -eq 25 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
elif [[ $SLURM_ARRAY_TASK_ID -eq 26 ]]
then
    python tools/test.py ../segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024/iter_160000.pth --work-dir ../work_dirs/segformer_mit-b5_8xb1-160k_cityscapes-512x1024 --cfg-options model.perform_attack=True model.attack_cfg.name='apgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.01 model.attack_cfg.epsilon=0.2509803922 model.attack_cfg.iterations=20
else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo Runtime: $runtime