CUDA_VISIBLE_DEVICES=7 \
python main_single_gpu.py \
-cfg='./configs/FF_Base.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./linear_base'
