CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/vip_s7.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='/dataset/imagenet' \
