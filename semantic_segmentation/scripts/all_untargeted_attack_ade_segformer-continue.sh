#!/usr/bin/env bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=all_untargeted_attack_ade_segformer-continue
#SBATCH --output=slurm/all_untargeted_attack_ade_segformer-continue%a.out
#SBATCH --array=5-6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

start=$(date +%s)

cd mmsegmentation
if [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]
then
    python tools/test.py ./configs/segformer/segformer_mit-b4_8xb2-160k_ade20k-512x512.py ../aa_workdir/ade/MIT-B4/segformer_mit-b4_512x512_160k_ade20k_20210728_183055-7f509d7d.pth --work-dir ../aa_workdir/ade/MIT-B4 --cfg-options model.perform_attack=True model.attack_cfg.name='segpgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.0000392157 model.attack_cfg.epsilon=64 model.attack_cfg.iterations=20   
elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]
then
    python tools/test.py ./configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py ../aa_workdir/ade/MIT-B5/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth --work-dir ../aa_workdir/ade/MIT-B5 --cfg-options model.perform_attack=True model.attack_cfg.name='segpgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.0000392157 model.attack_cfg.epsilon=4 model.attack_cfg.iterations=20
    python tools/test.py ./configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py ../aa_workdir/ade/MIT-B5/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth --work-dir ../aa_workdir/ade/MIT-B5 --cfg-options model.perform_attack=True model.attack_cfg.name='segpgd' model.attack_cfg.norm='linf' model.attack_cfg.alpha=0.0000392157 model.attack_cfg.epsilon=8 model.attack_cfg.iterations=20
    python tools/test.py ./configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py ../aa_workdir/ade/MIT-B5/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth --work-dir ../aa_workdir/ade/MIT-B5 --cfg-options model.perform_attack=True model.attack_cfg.name='segpgd' model.attack_cfg.norm='l2' model.attack_cfg.alpha=0.0000392157 model.attack_cfg.epsilon=64 model.attack_cfg.iterations=20   
else
    echo "All submitted"
fi

end=$(date +%s)
runtime=$((end-start))

echo Runtime: $runtime