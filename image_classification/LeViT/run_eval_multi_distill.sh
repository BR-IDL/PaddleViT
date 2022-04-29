CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu_distill.py \
-cfg='./configs/levit_384.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-pretrained='./levit_384.pdparams' \
-eval \
-amp
