CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/styleformer_lsun.yaml' \
-dataset='lsun' \
-batch_size=128 \
-eval \
-pretrained='./lsun' \
-data_path='/workspace/gan_datasets/church_outdoor_train_lmdb'
