CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/styleformer_lsun.yaml' \
-dataset='lsun' \
-batch_size=128 \
-eval \
-pretrained='./lsun' \
-data_path='/workspace/gan_datasets/church_outdoor_train_lmdb'
