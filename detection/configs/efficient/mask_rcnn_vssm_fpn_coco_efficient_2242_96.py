_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

# train_dataloader = dict(batch_size=2) # as gpus=8

model = dict(
    backbone=dict(
        type='MM_EfficientVSSM',
        depths=(2, 2, 4, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        pretrained="ckpts/efficient_vmamba_small.ckpt",
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
        window_size=2
    ),
)


optim_wrapper = dict(type='AmpOptimWrapper')