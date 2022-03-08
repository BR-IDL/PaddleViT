#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#python main_multi_gpu_linearprobe.py \
GLOG_v=0 python3 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" main_multi_gpu_linearprobe.py \
-cfg='./configs/vit_base_patch16_224_linearprobe.yaml' \
-dataset='imagenet2012' \
-batch_size=512 \
-accum_iter=2 \  # orriginal effective batch_size = 512bs * 4nodes * 8gpus. So for 2 node, accum_iter should be 2
-data_path='/dataset/imagenet' \
-pretrained='./mae_pretrain_vit_base.pdparams' \
-amp \
