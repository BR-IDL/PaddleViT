CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/resmlp_24_224.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='/dataset/imagenet' \
-amp
