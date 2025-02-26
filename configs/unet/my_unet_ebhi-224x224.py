_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', '../_base_/datasets/my_ebhi_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/my_scheduler.py'
]
crop_size = (224, 224)
log_processor = dict(by_epoch=True)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor,
             test_cfg=dict(mode='whole'))
