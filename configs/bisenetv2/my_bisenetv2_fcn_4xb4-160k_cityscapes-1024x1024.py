_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/pascal_voc_leaf_shanyao_3labels.py',
    '../_base_/default_runtime_epoch.py', '../_base_/schedules/my_schedule_epoch.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)