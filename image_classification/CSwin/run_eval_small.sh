CUDA_VISIBLE_DEVICES=7 \
python main_single_gpu.py \
-cfg='./configs/cswin_small_224.yaml' \
-dataset='imagenet2012' \
-batch_size=128 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./cswin_small_224'
