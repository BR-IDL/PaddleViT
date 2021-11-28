CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
    -cfg='./configs/mobileformer_26m.yaml' \
    -dataset='imagenet2012' \
    -num_classes=1000 \
    -batch_size=256 \
    -image_size=224 \
    -data_path='/dataset/imagenet' \
    -output='./output'
