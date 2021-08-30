CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/styleformer_celeba.yaml' \
-dataset='celeba' \
-batch_size=128 \
-eval \
-pretrained='./celeba' \
-data_path='/workspace/gan_datasets/celeba/img_align_celeba'
