CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu_pretrain.py \
-cfg='./configs/vit_base_patch16_224_pretrain.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-amp
