#CUDA_VISIBLE_DEVICES=4,5,6,7 \
#python main_multi_gpu.py \
GLOG_v=0 python3 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" main_multi_gpu.py \
-cfg='./configs/replknet_31b.yaml' \
-dataset='imagenet2012' \
-batch_size=4 \
-data_path='/dataset/imagenet' \
#-amp
