#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#python main_multi_gpu_finetune.py \
GLOG_v=0 python3 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" main_multi_gpu_finetune.py \
-cfg='./configs/vit_base_patch16_224_finetune_single_node.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-pretrained='./mae_vit_base_patch16' \
-amp \
