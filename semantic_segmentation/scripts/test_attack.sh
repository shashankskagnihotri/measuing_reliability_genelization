cd ..

alpha=0.01
iterations=20
name='cospgd'
norm='linf'
epsilon=4

python mmsegmentation/tools/test.py \
    mmsegmentation/configs/deeplabv3/deeplabv3_r50-d8_4xb4-160k_ade20k-512x512.py \
    checkpoint_files/deeplabv3/deeplabv3_r50-d8_512x512_160k_ade20k_20200615_123227-5d0ee427.pth 
    --work-dir temp