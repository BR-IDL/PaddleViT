CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/pvtv2_b0.yaml' \
-dataset='coco' \
-batch_size=2 \
-data_path='/dataset/coco' \
-eval \
-pretrained='./pvtv2_b0_maskrcnn'
