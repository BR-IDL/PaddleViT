CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu_finetune.py \
-cfg='./configs/vit_base_patch16_224_finetune.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='./dataset/imagenet' \
-amp \
-pretrained='./output/train-20211203-14-42-46/MAE-Epoch-10-Loss-0'
