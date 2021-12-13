CUDA_VISIBLE_DEVICES=0,1 \
python main_multi_gpu_finetune.py \
-cfg='./configs/vit_base_patch16_224_finetune.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='/dataset/imagenet' \
#-amp \
