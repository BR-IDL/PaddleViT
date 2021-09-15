CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/pvtv2_b2.yaml' \
-dataset='coco' \
-batch_size=4 \
-data_path='/dataset/coco' \
-eval \
-pretrained='./pvtv2_b2_maskrcnn'
