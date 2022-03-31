CUDA_VISIBLE_DEVICES=0,1,2,3 \
#CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu_distill.py \
-cfg='./configs/xcit_large_24_p8_224.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-pretrained='./xcit_large_24_p8_224_distill.pdparams' \
-eval \
-amp
