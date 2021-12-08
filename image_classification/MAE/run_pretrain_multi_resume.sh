CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu_pretrain.py \
-cfg='./configs/vit_base_patch16_224_pretrain.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-resume='./output/' \
-last_epoch=216 \
-amp
