crop_size = (
    256,
    256,
)
data_preprocessor = dict(
    bgr_to_rgb=False,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        64,
        64,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '/scratch/hshang/moody/final_phase_submission/processed_data'
dataset_type = 'HRFDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=200, type='CheckpointHook'),
    logger=dict(interval=10, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
img_scale = (
    584,
    565,
)
load_from = '/scratch/hshang/moody/final_phase_submission/official-phase-mins-eth/TeamCode/src/configs_ckpts/original_pretrained_weights/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_256x256_40k_hrf_20211210_202032-59daf7a4.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=64,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=128,
        in_index=3,
        loss_decode=dict(
            class_weight=[
                0.5,
                10.0,
            ],
            loss_weight=0.4,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        num_convs=1,
        out_channels=2,
        type='FCNHead'),
    backbone=dict(
        act_cfg=dict(type='ReLU'),
        base_channels=64,
        conv_cfg=None,
        dec_dilations=(
            1,
            1,
            1,
            1,
        ),
        dec_num_convs=(
            2,
            2,
            2,
            2,
        ),
        downsamples=(
            True,
            True,
            True,
            True,
        ),
        enc_dilations=(
            1,
            1,
            1,
            1,
            1,
        ),
        enc_num_convs=(
            2,
            2,
            2,
            2,
            2,
        ),
        in_channels=3,
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=5,
        strides=(
            1,
            1,
            1,
            1,
            1,
        ),
        type='UNet',
        upsample_cfg=dict(type='InterpConv'),
        with_cp=False),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=16,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        dropout_ratio=0.1,
        in_channels=64,
        in_index=4,
        loss_decode=[
            dict(
                alpha=0.1,
                gamma=2.0,
                loss_name='loss_focal',
                loss_weight=1.0,
                type='FocalLoss',
                use_sigmoid=True),
            dict(loss_name='loss_dice', loss_weight=3.0, type='DiceLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        out_channels=2,
        type='ASPPHead'),
    pretrained=None,
    test_cfg=dict(crop_size=(
        64,
        64,
    ), mode='slide', stride=(
        42,
        42,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=40000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
randomness = dict(seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='splits/val.txt',
        data_prefix=dict(img_path='cropped_img', seg_map_path='cropped_masks'),
        data_root='/scratch/hshang/moody/final_phase_submission/processed_data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                320,
                240,
            ), type='Resize'),
            dict(size_divisor=16, type='ResizeToMultiple'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='HRFDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mDice',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        320,
        240,
    ), type='Resize'),
    dict(size_divisor=16, type='ResizeToMultiple'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=200, type='IterBasedTrainLoop', val_interval=200)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='splits/train.txt',
        data_prefix=dict(img_path='cropped_img', seg_map_path='cropped_masks'),
        data_root='/scratch/hshang/moody/final_phase_submission/processed_data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    1.5,
                ),
                scale=(
                    256,
                    256,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    256,
                    256,
                ), type='RandomCrop'),
            dict(size_divisor=16, type='ResizeToMultiple'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='HRFDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            1.5,
        ),
        scale=(
            256,
            256,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        256,
        256,
    ), type='RandomCrop'),
    dict(size_divisor=16, type='ResizeToMultiple'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='splits/val.txt',
        data_prefix=dict(img_path='cropped_img', seg_map_path='cropped_masks'),
        data_root='/scratch/hshang/moody/final_phase_submission/processed_data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                320,
                240,
            ), type='Resize'),
            dict(size_divisor=16, type='ResizeToMultiple'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='HRFDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mDice',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/scratch/hshang/moody/final_phase_submission/mmsegunet_model'
