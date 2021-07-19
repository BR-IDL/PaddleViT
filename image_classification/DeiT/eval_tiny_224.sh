CUDA_VISIBLE_DEVICES=7 \
python main_single_gpu.py \
-cfg='./config/deit_tiny_distilled_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=128 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./deit_tiny_patch16_224'
