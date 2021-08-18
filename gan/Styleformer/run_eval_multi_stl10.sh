CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/styleformer_stl10.yaml' \
-dataset='stl10' \
-batch_size=128 \
-eval \
-pretrained='./stl10' \
-data_path='/workspace/gan_datasets/stl10_binary'
