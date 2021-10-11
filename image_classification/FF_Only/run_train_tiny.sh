CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/ff_tiny.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='/dataset/imagenet' \
-amp
