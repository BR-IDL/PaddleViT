CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/cait_xxs24_224.yaml' \
-dataset='imagenet2012' \
-batch_size=4 \
-data_path='/dataset/imagenet' \
-amp
