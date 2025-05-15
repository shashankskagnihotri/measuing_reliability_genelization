_base_ = [
    '../../mmsegmentation/configs/_base_/models/fcn_unet_s5-d16.py', '../../mmsegmentation/configs/_base_/datasets/ade20k.py',
    '../../mmsegmentation/configs/_base_/default_runtime.py', '../../mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]

crop_size = (64, 64)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
train_dataloader = dict(batch_size=1, num_workers=16)
val_dataloader = dict(batch_size=1, num_workers=16)
test_dataloader = val_dataloader

# The original problem comes from the validation set. You also need to specifiy that it has the correct size


