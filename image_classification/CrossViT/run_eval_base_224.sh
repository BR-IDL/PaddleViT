CUDA_VISIBLE_DEVICES=0 \
python eval.py \
-cfg='./configs/pd_crossvit_base_224.yaml' \
-dataset='imagenet2012' \
-batch_size=128 \
-data_path='/media/misa/软件/CrossViT复现日记/ImageNetDataset/ILSVRC2012_img_val' \
-eval  \
-pretrained='./port_weights/pd_crossvit_base_224.pdparams'
