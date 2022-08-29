#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GLOG_v=0 python3 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" main_multi_gpu.py \
-cfg='./configs/mobileone_s0.yaml' \
-dataset='imagenet2012' \
-batch_size=128 \
-data_path='/dataset/imagenet' \
-amp \
-pretrained='./s0.pdparams' \
-eval
