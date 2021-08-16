CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/swin_base_patch4_window7_224.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./swin_base_patch4_window7_224' \
