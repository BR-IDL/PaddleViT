CUDA_VISIBLE_DEVICES=7 \
python main_single_gpu.py \
-cfg='./configs/gmlp_s16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./gmlp_s16_224'
