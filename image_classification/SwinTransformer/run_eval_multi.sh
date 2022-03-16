CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/swin_large_patch4_window12_384.yaml' \
-dataset='imagenet2012' \
-batch_size=64 \
-data_path='/dataset/imagenet' \
-pretrained='./swin_large_patch4_window12_384.pdparams' \
-eval \
-amp
