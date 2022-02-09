_base_="base.py"
data_root = 'data/dent/'
classes = ['dent']
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=8,
    train=dict(

        sup=dict(

            ann_file=data_root+"annotations/train.json",
            img_prefix=data_root+"images/",
            classes=classes

        ),
        unsup=dict(

            ann_file=data_root+"annotations/valid.json",
            img_prefix=data_root+"images/",
            classes=classes

        ),
    ),
    val=dict(

            ann_file=data_root+"annotations/test.json",
            img_prefix=data_root+"images/",
            classes=classes

    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)
work_dir = './work_dirs/dent'

