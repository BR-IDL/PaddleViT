CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/pvtv2_b1.yaml' \
-dataset='coco' \
-batch_size=4 \
-data_path='/dataset/coco' \
-eval \
-pretrained='./pvtv2_b1_maskrcnn'
