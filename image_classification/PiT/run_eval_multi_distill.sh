CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu_distill.py \
-cfg='./configs/pit_xs_distilled_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-pretrained='./pit_xs_distilled_224.pdparams' \
-eval \
-amp
