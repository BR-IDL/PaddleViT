CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/cyclemlp_b5.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./cyclemlp_b5'
