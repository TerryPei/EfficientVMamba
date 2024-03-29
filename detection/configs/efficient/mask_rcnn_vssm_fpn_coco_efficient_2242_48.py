_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]
import os


model = dict(
    backbone=dict(
        type='MM_EfficientVSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="ckpts/efficient_vmamba_tiny.ckpt",
        depths=(2, 2, 9, 2),
        dims=48,
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
        window_size=2
    ),
)

# train_dataloader = dict(batch_size=2) # as gpus=8
train_dataloader = dict(batch_size=4) # as gpus=4


optim_wrapper = dict(type='AmpOptimWrapper')