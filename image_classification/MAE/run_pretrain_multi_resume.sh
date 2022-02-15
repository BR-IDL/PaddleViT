CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu_pretrain.py \
-cfg='./configs/vit_base_patch16_224_pretrain_dec1.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-resume='./output/train-20220125-17-48-06/PRETRAIN-Epoch-99-Loss-0.5566961133140487' \
-last_epoch=99 \
-amp
