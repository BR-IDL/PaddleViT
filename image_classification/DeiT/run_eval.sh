CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/deit_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=16 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./deit_base_distilled_patch16_224' \
