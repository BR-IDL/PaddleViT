CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/focal_tiny_patch4_window7_224.yaml' \
    -dataset='imagenet2012' \
    -num_classes=1000 \
    -batch_size=64 \
    -image_size=224 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./focal_tiny_patch4_window7_224'