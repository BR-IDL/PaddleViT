CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/convmlp_l.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./convmlp_l'
