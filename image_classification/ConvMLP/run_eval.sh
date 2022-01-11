CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/convmlp_s.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./convmlp_s'
