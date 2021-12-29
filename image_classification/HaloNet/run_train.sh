CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/halonet_26t_256.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
