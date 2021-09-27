CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg="./configs/cyclemlp_b4.yaml" \
-dataset="imagenet2012" \
-batch_size=128 \
-data_path="/dataset/imagenet" \
-eval \
-pretrained="./cyclemlp_b4"
