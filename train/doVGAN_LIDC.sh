CUDA_VISIBLE_DEVICES=1 PL_TORCH_DISTRIBUTED_BACKEND=gloo python train_vqgan.py \
    dataset=lidc dataset.root_dir=/datastd/LIDC/manifest-1600709154662/LIDC-IDRI/ \
    model=vq_gan_3d \
    model.gpus=1 \
    model.default_root_dir_postfix='flair' \
    model.precision=16 \
    model.embedding_dim=8 \
    model.n_hiddens=16 \
    model.downsample=[2,2,2] \
    model.num_workers=6 \
    model.gradient_clip_val=1.0 \
    model.lr=3e-4 \
    model.discriminator_iter_start=10000 \
    model.perceptual_weight=4 \
    model.image_gan_weight=1 \
    model.video_gan_weight=1 \
    model.gan_feat_weight=4 \
    model.batch_size=2 \
    model.n_codes=16384 \
    model.accumulate_grad_batches=1 