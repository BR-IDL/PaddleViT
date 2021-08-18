CUDA_VISIBLE_DEVICES=2 \
python main_single_gpu.py \
-cfg='./configs/styleformer_celeba.yaml' \
-dataset='celeba' \
-batch_size=64 \
-eval \
-pretrained='./celeba' \
-data_path='/workspace/gan_datasets/celeba/img_align_celeba'
