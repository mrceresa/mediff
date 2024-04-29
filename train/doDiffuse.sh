CUDA_VISIBLE_DEVICES=0 python train_ddpm.py \
    model=ddpm dataset=brats \
    model.results_folder_postfix='flair' \
    model.vqgan_ckpt='/mnt/data/projects/jrc/brics/medical_diffusion_checkpoints/vq_gan/BRATS/flair/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt' \
    model.diffusion_img_size=32 \
    model.diffusion_depth_size=32 \
    model.diffusion_num_channels=8 \
    model.dim_mults=[1,2,4,8] \
    model.batch_size=5 \
    model.gpus=0

#model.load_milestone='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/ddpm/MRNet/model-8.pt'