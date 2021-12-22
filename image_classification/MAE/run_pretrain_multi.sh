CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
python main_multi_gpu_pretrain.py \
-cfg='./configs/vit_base_patch16_224_pretrain_dec1.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='/dataset/imagenet' \
-mae_pretrain \
#-amp
