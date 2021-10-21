CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_single_gpu.py \
    -cfg='./configs/focal_tiny_patch4_window7_224.yaml' \
    -dataset='imagenet2012' \
    -num_classes=1000 \
    -batch_size=4 \
    -image_size=224 \
    -data_path='/dataset/imagenet' \
    -output='./output'