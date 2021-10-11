CUDA_VISIBLE_DEVICES=7 \
python main_single_gpu.py \
-cfg='./configs/mixer_b16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='/dataset/imagenet' \
-amp
