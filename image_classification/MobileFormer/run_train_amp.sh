CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/mobileformer_26m.yaml' \
-dataset='imagenet2012' \
-num_classes=1000 \
-batch_size=4 \
-image_size=224 \
-data_path='/dataset/imagenet' \
-output='./output' \
-amp