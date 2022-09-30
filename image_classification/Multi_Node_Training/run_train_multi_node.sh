CUDA_VISIBLE_DEVICES=0,1 \
python main_multi_node.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='./dataset/imagenet' \
-ips='172.18.0.2, 172.18.0.3' # the ips should be replaced
#-amp
