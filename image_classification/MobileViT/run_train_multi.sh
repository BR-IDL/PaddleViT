CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/mobilevit_xxs.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-amp \
#-multi_scale_sampler_ddp
