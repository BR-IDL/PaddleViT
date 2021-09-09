CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/repmlpres50_light_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./RepMLP-Res50-light-224_train'
