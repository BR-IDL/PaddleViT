CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/styleformer_lsun.yaml' \
-dataset='lsun' \
-batch_size=32 \
-eval \
-pretrained='./lsun' \
-data_path='../church_outdoor_train_lmdb'
