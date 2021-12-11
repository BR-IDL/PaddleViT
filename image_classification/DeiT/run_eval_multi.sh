#CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/deit_tiny_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./deit_tiny_distilled_patch16_224' \
-ngpus=4
