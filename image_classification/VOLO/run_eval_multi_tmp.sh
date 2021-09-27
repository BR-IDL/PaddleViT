#CUDA_VISIBLE_DEVICES=4,5,6,7 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/volo_d1_224.yaml' \
-dataset='imagenet2012' \
-batch_size=64 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./d1_224_84.2' \
