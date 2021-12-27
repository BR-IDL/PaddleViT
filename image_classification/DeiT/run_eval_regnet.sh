CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_eval_regnet_multi_gpu.py \
-cfg='./configs/regnety_160.yaml' \
-dataset='imagenet2012' \
-batch_size=4 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./regnety_160' \
-ngpus=4
