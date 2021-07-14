CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/cait_m48_448.yaml' \
-dataset='imagenet2012' \
-batch_size=1 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./cait_m48_448' \
-ngpus=4
