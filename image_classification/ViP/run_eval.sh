CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/vip_m7.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./vip_m7'
