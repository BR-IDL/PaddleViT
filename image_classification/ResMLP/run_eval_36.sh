CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/resmlp_36_224.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./resmlp_36_224' \
