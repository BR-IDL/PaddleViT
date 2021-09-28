CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/beit_base_patch16_384.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./beit_base_patch16_384_ft22kto1k'
