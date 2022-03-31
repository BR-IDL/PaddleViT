CUDA_VISIBLE_DEVICES=0,1 \
python main_multi_gpu_distill.py \
-cfg='./configs/xcit_nano_12_p8_224.yaml' \
-dataset='imagenet2012' \
-batch_size=16 \
-data_path='/dataset/imagenet' \
-teacher_model_path='./regnety_160.pdparams' \
-amp
