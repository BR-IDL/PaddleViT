CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/ff_base.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./linear_base'
