_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]
import os


model = dict(
    backbone=dict(
        type='MM_EfficientVSSM',
        depths=(2, 2, 9, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        pretrained="ckpts/efficient_vmamba_base.ckpt",
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



# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

# optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }),
#     optimizer=dict(
#         _delete_=True,
#         type='AdamW',
#         lr=0.0001,
#         betas=(0.9, 0.999),
#         weight_decay=0.05))
optim_wrapper = dict(type='AmpOptimWrapper')

