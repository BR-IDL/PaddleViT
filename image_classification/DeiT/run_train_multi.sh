CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/deit_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=4 \
-data_path='/dataset/imagenet' \
-teacher_model='./regnety_160' \
-amp
#-pretrained='./deit_base_distilled_patch16_224'
