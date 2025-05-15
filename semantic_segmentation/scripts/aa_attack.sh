#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=aa_attack_segformer
#SBATCH --output=slurm/aa_attack_segformer.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

cd mmsegmentation

# epsilon=2
# alpha=0.01
# name='cospgd'
# iterations=5
# norm='linf'

# with output
# python tools/test.py ./configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py ../aa_workdir/ade/MIT-B0/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth --cfg-options data_preprocessor.enable_normalization=False model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/ade/MIT-B0/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha} --out ../aa_workdir/ade/MIT-B0/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/output --show-dir ../aa_workdir/ade/MIT-B0/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}/show_dir

# without output
# python tools/test.py ./configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py ../aa_workdir/ade/MIT-B0/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/ade/MIT-B0/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}

# all segformer mit b0
alpha=0.01
iterations=20
name='cospgd'

norm='linf'
epsilon=4
python tools/test.py ./configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py ../aa_workdir/ade/MIT-B0/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/ade/MIT-B0/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}

norm='linf'
epsilon=8
python tools/test.py ./configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py ../aa_workdir/ade/MIT-B0/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/ade/MIT-B0/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}

norm="l2"
epsilon=64
python tools/test.py ./configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py ../aa_workdir/ade/MIT-B0/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth --cfg-options model.perform_attack=True model.attack_cfg.name=${name} model.attack_cfg.norm=${norm} model.attack_cfg.alpha=${alpha} model.attack_cfg.epsilon=${epsilon} model.attack_cfg.iterations=${iterations} --work-dir ../aa_workdir/ade/MIT-B0/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}

# adv mmsegmentation:
# python tools/test.py configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py ../benchmarking_robustness/semantic_segmentation/aa_workdir/ade/MIT-B0/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth --perform_attack --attack ${name} --iterations ${iterations} --epsilon ${epsilon} --alpha ${alpha} --norm ${norm} --work-dir adv_workdir/ade/MIT-B0/attack_${name}/norm_${norm}/iterations_${iterations}/epsilon_${epsilon}/alpha_${alpha}

end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime