CUDA_VISIBLE_DEVICES=0 \
python main_multi_node.py \
-cfg='./configs/cyclemlp_b1.yaml' \
-dataset='imagenet2012' \
-batch_size=8 \
-data_path='/dataset/imagenet' \
-ips='172.18.0.2, 172.18.0.3, 172.18.0.4, 172.18.0.5' # the ips should be replaced
#-amp
