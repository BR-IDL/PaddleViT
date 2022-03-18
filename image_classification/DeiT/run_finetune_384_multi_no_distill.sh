CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu_no_distill.py \
-cfg='./configs/deit_base_patch16_384.yaml' \
-dataset='imagenet2012' \
-batch_size=16 \
-data_path='/dataset/imagenet' \
-pretrained='./deit_base_patch16_224.pdparams' \
-amp
