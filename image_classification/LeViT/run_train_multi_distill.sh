CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu_distill.py \
-cfg='./configs/levit_128s.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-teacher_mdoel_path='./regnety_160.pdparams' \
-amp
