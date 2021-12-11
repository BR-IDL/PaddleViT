CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/deit_tiny_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=200 \
-data_path='/dataset/imagenet' \
-teacher_model='./regnety_160' \
-amp \
#-pretrained='./deit_base_distilled_patch16_224'
