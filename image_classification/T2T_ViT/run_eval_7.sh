CUDA_VISIBLE_DEVICES=7 \
python main_single_gpu.py \
-cfg='./configs/t2t_vit_7.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./t2t_vit_7'
