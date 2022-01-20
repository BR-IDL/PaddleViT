CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python main_dino_multi_gpu.py \
    -cfg="./configs/vit_small_patch16_224.yaml" \
    -dataset="imagenet2012" \
    -batch_size=32 \
    -data_path="/dataset/imagenet" \
    -amp \
