
# Segmentation toolkit based on ViT

## Quick start: training and testing models

### 1. Preparing data

 Download Pascal-Context dataset. It should have this basic structure:  

pascal_context
|-- Annotations
|-- ImageSets
|-- JPEGImages
|-- SegmentationClass
|-- SegmentationClassContext
|-- SegmentationObject
|-- trainval_merged.json
|-- voc2010_to_pascalcontext.py

### 2. Testing
#### Single-scale testing on single GPU

```shell
CUDA_VISIBLE_DEVICES=0 python3  val.py  \
    --config ./configs/SETR/SETR_MLA_Large_480x480_80k_pascal_context_bs_8.yaml \
    --model_path ./pretrain_models/setr/SETR_MLA_pascal_context_b8_80k.pdparams
```
> Note:
> - The `-model_path` option accepts the path of pretrained weights file (segmentation model, e.g., setr).

#### Single-scale testing on multi GPU

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u -m paddle.distributed.launch val.py \
    --config ./configs/SETR/SETR_MLA_Large_480x480_80k_pascal_context_bs_8.yaml \
    --model_path ./pretrain_models/setr/SETR_MLA_pascal_context_b8_80k.pdparams
```
> Note:
>
> - that the `-pretrained` option accepts the path of pretrained weights file (segmentation model, e.g., setr)


### 3. Training
#### Training on single GPU

```shell
CUDA_VISIBLE_DEVICES=0 python3  train.py \
    --config ./configs/SETR/SETR_MLA_Large_480x480_80k_pascal_context_bs_8.yaml
```
> Note:
> - The training options such as lr, image size, model layers, etc., can be changed in the `.yaml` file set in `-cfg`. All the available settings can be found in `./config.py`

#### Training on multi GPU

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u -m paddle.distributed.launch train.py \
    --config ./configs/SETR/SETR_MLA_Large_480x480_80k_pascal_context_bs_8.yaml

```
> Note:
>
> - The training options such as lr, image size, model layers, etc., can be changed in the `.yaml` file set in `-cfg`. All the available settings can be found in `./config.py`


