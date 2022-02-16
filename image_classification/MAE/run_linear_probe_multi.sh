CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu_linearprobe.py \
-cfg='./configs/vit_base_patch16_224_linearprobe.yaml' \
-dataset='imagenet2012' \
-batch_size=512 \
-data_path='/dataset/imagenet' \
-amp \
-pretrained='./mae_pretrain_vit_base'
