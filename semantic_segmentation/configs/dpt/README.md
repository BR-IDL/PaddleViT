# Vision Transformers for Dense Prediction, [arxiv](https://arxiv.org/pdf/2103.13413.pdf) 

The official pytorch implementation is [here](https://github.com/isl-org/DPT).
## Framework
<img src="../../figure/dpt_framework.png" alt="drawing" width="100%" height="100%"/>

## Model Zoo ##
### ADE20K ###
|Model      | Backbone  | Batch_size | Iteration | mIoU (ss) | mIoU (ms+flip) | Backbone_checkpoint | Model_checkpoint     |     ConfigFile  |
|-----------|-----------|------------|-----------|-----------|----------------|-----------------------------------------------|-----------------------------------------------------------------------|------------|
|DPT        | ViT_Large |     16     |     160k   |  47.21   |       -        |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)      |[baidu](https://pan.baidu.com/s/1PCSC1Kvcg291gqp6h5pDCg)(ts7h)   |  [config](semantic_segmentation/configs/dpt/DPT_Large_480x480_160k_ade20k_bs_16.yaml)

## Reference
```
@article{ranftl2021vision,
  title={Vision transformers for dense prediction},
  author={Ranftl, Ren{\'e} and Bochkovskiy, Alexey and Koltun, Vladlen},
  journal={arXiv preprint arXiv:2103.13413},
  year={2021}
}
```
