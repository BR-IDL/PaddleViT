CUDA_VISIBLE_DEVICES=7 \
python main_single_gpu.py \
-cfg='./configs/mixer_l16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=16 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./mixer_l16_224'
