CUDA_VISIBLE_DEVICES=0,1 \
python main_multi_gpu_linearprobe.py \
-cfg='./configs/vit_base_patch16_224_linearprobe.yaml' \
-dataset='imagenet2012' \
-batch_size=2 \
-data_path='/dataset/imagenet' \
-amp \
-pretrained='./output/train-20220125-17-48-06/PRETRAIN-Epoch-99-Loss-0.5566961133140487'
