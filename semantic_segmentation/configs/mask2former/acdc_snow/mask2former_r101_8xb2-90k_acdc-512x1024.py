_base_ = ['./mask2former_r50_8xb2-90k_acdc-512x1024.py']

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
