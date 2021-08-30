CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/styleformer_stl10.yaml' \
-dataset='stl10' \
-batch_size=128 \
-eval \
-pretrained='./stl10' \
-data_path='/workspace/gan_datasets/stl10_binary'
