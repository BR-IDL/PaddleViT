#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CUDA_VISIBLE_DEVICES=0,1 \
python main_multi_gpu_no_distill.py \
-cfg='./configs/xcit_nano_12_p8_224.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='/dataset/imagenet' \
-amp
