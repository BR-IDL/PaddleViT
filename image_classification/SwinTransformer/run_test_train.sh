CUDA_VISIBLE_DEVICES=0,1 \
python main_multi_gpu.py \
-cfg='./configs/swin_tiny_patch4_window7_224.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='/dataset/imagenet' \
-amp
