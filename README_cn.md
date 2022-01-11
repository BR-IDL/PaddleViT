简体中文 | [English](./README.md)

# PaddlePaddle Vision Transformers #

[![GitHub](https://img.shields.io/github/license/BR-IDL/PaddleViT?color=blue)](./LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/br-idl/paddlevit/badge)](https://www.codefactor.io/repository/github/br-idl/paddlevit)
[![CLA assistant](https://cla-assistant.io/readme/badge/BR-IDL/PaddleViT)](https://cla-assistant.io/BR-IDL/PaddleViT)
[![GitHub Repo stars](https://img.shields.io/github/stars/BR-IDL/PaddleViT?style=social)](https://github.com/BR-IDL/PaddleViT/stargazers)


<p align="center">    
    <img src="./PaddleViT.png" width="100%"/>
</p>
 
## State-of-the-art Visual Transformer and MLP Models for PaddlePaddle ##

:robot: PaddlePaddle Visual Transformers (`PaddleViT` 或 `PPViT`) 为开发者提供视觉领域的高性能Transformer模型实现。 我们的主要实现基于Visual Transformers, Visual Attentions, 以及 MLPs等视觉模型算法。 此外，PaddleViT集成了PaddlePaddle 2.1+中常用的layers, utilities, optimizers, schedulers, 数据增强, 以及训练/评估脚本等。我们持续关注SOTA的ViT和MLP模型算法，并提供完整训练、测试代码。PaddleViT的核心任务是**为用户提供方便易用的CV领域前沿算法**。

:robot: PaddleViT 为多项视觉任务提供模型和工具，例如图像分类，目标检测，语义分割，GAN等。每个模型架构均在独立的Python模块中定义，以便于用户能够快速的开展研究和进行实验。同时，我们也提供了模型的预训练权重文件，以便您加载并使用自己的数据集进行微调。PaddleViT还集成了常用的工具和模块，用于自定义数据集、数据预处理，性能评估以及分布式训练等。

:robot: PaddleViT 基于深度学习框架 [PaddlePaddle](https://www.paddlepaddle.org/)进行开发, 我们同时在[Paddle AI Studio](https://aistudio.baidu.com/aistudio/index)上提供了项目教程(coming soon). 对于新用户能够简单易操作。


## 视觉任务 ##
PaddleViT 提供了多项视觉任务的模型和工具，请访问以下链接以获取详细信息： 
- [PaddleViT-Cls](./image_classification) 用于 图像分类
- [PaddleViT-Det](./object_detection/DETR) 用于 目标检测
- [PaddleViT-Seg](./semantic_segmentation) 用于 语义分割
- [PaddleViT-GAN](./gan) 用于 生成对抗模型
  
我们同时提供对应教程：
- Notebooks (即将更新)
- Online Course (即将更新)

## 项目特性 ##
1. **SOTA模型的完整实现**
   - 提供多项CV任务的SOTA Transformer 模型 
   - 提供高性能的数据处理和训练方法
   - 持续推出最新的SOTA算法的实现

2. **易于使用的工具**
   - 通过简单配置即可实现对模型变体的实现
   - 将实用功能与工具进行模块化设计
   - 对于教育者和从业者的使用低门槛
   - 所有模型以统一框架实现

3. **符合用户的自定义需求**
   - 提供每个模型的实现的最佳实践
   - 提供方便用户调整自定义配置的模型实现
   - 模型文件可以独立使用以便于用户快速复现算法

4. **高性能**
   - 支持DDP (多进程训练/评估，其中每个进程在单个GPU上运行)
   - 支持混合精度 support (AMP)训练策略
  

  
## ViT模型算法 ##

### 图像分类 (Transformers) ###
1. **[ViT](./image_classification/ViT)** (from Google), released with paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
2. **[DeiT](./image_classification/DeiT)** (from Facebook and Sorbonne), released with paper [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877), by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.
3. **[Swin Transformer](./image_classification/SwinTransformer)** (from Microsoft), released with paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030), by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
4. **[VOLO](./image_classification/VOLO)** (from Sea AI Lab and NUS), released with paper [VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/abs/2106.13112), by Li Yuan, Qibin Hou, Zihang Jiang, Jiashi Feng, Shuicheng Yan.
5. **[CSwin Transformer](./image_classification/CSwin)** (from USTC and Microsoft), released with paper [CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows
](https://arxiv.org/abs/2107.00652), by Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, Baining Guo.
6. **[CaiT](./image_classification/CaiT)** (from Facebook and Sorbonne), released with paper [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239), by Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, Hervé Jégou.
7. **[PVTv2](./image_classification/PVTv2)** (from NJU/HKU/NJUST/IIAI/SenseTime), released with paper [PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797), by Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao.
8. **[Shuffle Transformer](./image_classification/Shuffle_Transformer)** (from Tencent), released with paper [Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer](https://arxiv.org/abs/2106.03650), by Zilong Huang, Youcheng Ben, Guozhong Luo, Pei Cheng, Gang Yu, Bin Fu.
9. **[T2T-ViT](./image_classification/T2T_ViT)** (from NUS and YITU), released with paper [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet
](https://arxiv.org/abs/2101.11986), by Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zihang Jiang, Francis EH Tay, Jiashi Feng, Shuicheng Yan.
10. **[CrossViT](./image_classification/CrossViT)** (from IBM), released with paper [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://arxiv.org/abs/2103.14899), by Chun-Fu Chen, Quanfu Fan, Rameswar Panda.
11. **[BEiT](./image_classification/BEiT)** (from Microsoft Research), released with paper [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254), by Hangbo Bao, Li Dong, Furu Wei.
12. **[Focal Transformer](./image_classification/Focal_Transformer)** (from Microsoft), released with paper [Focal Self-attention for Local-Global Interactions in Vision Transformers](https://arxiv.org/abs/2107.00641), by Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan and Jianfeng Gao.
13. **[Mobile-ViT](./image_classification/MobileViT)** (from Apple), released with paper [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178), by Sachin Mehta, Mohammad Rastegari.
14. **[ViP](./image_classification/ViP)** (from National University of Singapore), released with [Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition](https://arxiv.org/abs/2106.12368), by Qibin Hou and Zihang Jiang and Li Yuan and Ming-Ming Cheng and Shuicheng Yan and Jiashi Feng.
15. **[XCiT](./image_classification/XCiT)** (from Facebook/Inria/Sorbonne), released with paper [XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681), by Alaaeldin El-Nouby, Hugo Touvron, Mathilde Caron, Piotr Bojanowski, Matthijs Douze, Armand Joulin, Ivan Laptev, Natalia Neverova, Gabriel Synnaeve, Jakob Verbeek, Hervé Jegou.
16. **[PiT](./image_classification/PiT)** (from NAVER/Sogan University), released with paper [Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302), by Byeongho Heo, Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, Junsuk Choe, Seong Joon Oh.
17. **[HaloNet](./image_classification/HaloNet)**, (from Google), released with paper [Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/abs/2103.12731), by Ashish Vaswani, Prajit Ramachandran, Aravind Srinivas, Niki Parmar, Blake Hechtman, Jonathon Shlens.
18. **[PoolFormer](./image_classification/PoolFormer)**, (from Sea AI Lab/NUS), released with paper [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418), by Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, Shuicheng Yan.
19. **[BoTNet](./image_classification/BoTNet)**, (from UC Berkeley/Google), released with paper [Bottleneck Transformers for Visual Recognition](https://arxiv.org/abs/2101.11605), by Aravind Srinivas, Tsung-Yi Lin, Niki Parmar, Jonathon Shlens, Pieter Abbeel, Ashish Vaswani.
20. **[CvT](./image_classification/CvT)** (from McGill/Microsoft), released with paper [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808), by Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang
21. **[HvT](./image_classification/HVT)** (from Monash University), released with paper [Scalable Vision Transformers with Hierarchical Pooling](https://arxiv.org/abs/2103.10619), by Zizheng Pan, Bohan Zhuang, Jing Liu, Haoyu He, Jianfei Cai.



### 图像分类 (MLP & others) ###
1. **[MLP-Mixer](./image_classification/MLP-Mixer)** (from Google), released with paper [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601), by Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy
2. **[ResMLP](./image_classification/ResMLP)** (from Facebook/Sorbonne/Inria/Valeo), released with paper [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404), by Hugo Touvron, Piotr Bojanowski, Mathilde Caron, Matthieu Cord, Alaaeldin El-Nouby, Edouard Grave, Gautier Izacard, Armand Joulin, Gabriel Synnaeve, Jakob Verbeek, Hervé Jégou.
3. **[gMLP](./image_classification/gMLP)** (from Google), released with paper [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050), by Hanxiao Liu, Zihang Dai, David R. So, Quoc V. Le.
4. **[FF Only](./image_classification/FF_Only)** (from Oxford), released with paper [Do You Even Need Attention? A Stack of Feed-Forward Layers Does Surprisingly Well on ImageNet](https://arxiv.org/abs/2105.02723), by Luke Melas-Kyriazi.
5. **[RepMLP](./image_classification/RepMLP)** (from BNRist/Tsinghua/MEGVII/Aberystwyth), released with paper [RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition](https://arxiv.org/abs/2105.01883), by Xiaohan Ding, Chunlong Xia, Xiangyu Zhang, Xiaojie Chu, Jungong Han, Guiguang Ding.
6. **[CycleMLP](./image_classification/CycleMLP)** (from HKU/SenseTime), released with paper [CycleMLP: A MLP-like Architecture for Dense Prediction](https://arxiv.org/abs/2107.10224), by Shoufa Chen, Enze Xie, Chongjian Ge, Ding Liang, Ping Luo.
7. **[ConvMixer](./image_classification/ConvMixer)** (from Anonymous), released with [Patches Are All You Need?](https://openreview.net/forum?id=TVHS5Y4dNvM), by Anonymous.
8. **[ConvMLP](./image_classification/ConvMLP)** (from UO/UIUC/PAIR), released with [ConvMLP: Hierarchical Convolutional MLPs for Vision](https://arxiv.org/abs/2109.04454), by Jiachen Li, Ali Hassani, Steven Walton, Humphrey Shi.


#### 即将更新: ####
1. **[DynamicViT]()** (from Tsinghua/UCLA/UW), released with paper [DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification](https://arxiv.org/abs/2106.02034), by Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, Cho-Jui Hsieh.



### 目标检测 ###
1. **[DETR](./object_detection/DETR)** (from Facebook), released with paper [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko.
2. **[Swin Transformer](./object_detection/Swin)** (from Microsoft), released with paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030), by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
3. **[PVTv2](./object_detection/PVTv2)** (from NJU/HKU/NJUST/IIAI/SenseTime), released with paper [PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797), by Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao.

#### 即将更新: ####
1. **[Focal Transformer]()** (from Microsoft), released with paper [Focal Self-attention for Local-Global Interactions in Vision Transformers](https://arxiv.org/abs/2107.00641), by Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan and Jianfeng Gao.
2. **[UP-DETR]()** (from Tencent), released with paper [UP-DETR: Unsupervised Pre-training for Object Detection with Transformers](https://arxiv.org/abs/2011.09094), by Zhigang Dai, Bolun Cai, Yugeng Lin, Junying Chen.




### 目标分割 ###
#### 现有模型: ####
1. **[SETR](./semantic_segmentation)** (from Fudan/Oxford/Surrey/Tencent/Facebook), released with paper [Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers](https://arxiv.org/abs/2012.15840), by Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip H.S. Torr, Li Zhang.
2. **[DPT](./semantic_segmentation)** (from Intel), released with paper [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413), by René Ranftl, Alexey Bochkovskiy, Vladlen Koltun.
3. **[Swin Transformer](./semantic_segmentation)** (from Microsoft), released with paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030), by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
2. **[Segmenter](./semantic_segmentation)** (from Inria), realeased with paper [Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/pdf/2105.05633.pdf), by Robin Strudel, Ricardo Garcia, Ivan Laptev, Cordelia Schmid.
3. **[Trans2seg](./semantic_segmentation)** (from HKU/Sensetime/NJU), released with paper [Segmenting Transparent Object in the Wild with Transformer](https://arxiv.org/pdf/2101.08461.pdf), by Enze Xie, Wenjia Wang, Wenhai Wang, Peize Sun, Hang Xu, Ding Liang, Ping Luo.
4. **[SegFormer](./semantic_segmentation)** (from HKU/NJU/NVIDIA/Caltech), released with paper [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203), by Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo.
5. **[CSwin Transformer]()** (from USTC and Microsoft), released with paper [CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows

#### 即将更新:  ####
1. **[FTN]()** (from Baidu), released with paper [Fully Transformer Networks for Semantic Image Segmentation](https://arxiv.org/pdf/2106.04108.pdf), by Sitong Wu, Tianyi Wu, Fangjian Lin, Shengwei Tian, Guodong Guo.
2. **[Shuffle Transformer]()** (from Tencent), released with paper [Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer](https://arxiv.org/abs/2106.03650), by Zilong Huang, Youcheng Ben, Guozhong Luo, Pei Cheng, Gang Yu, Bin Fu
3. **[Focal Transformer]()** (from Microsoft), released with paper [Focal Self-attention for Local-Global Interactions in Vision Transformers](https://arxiv.org/abs/2107.00641), by Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan and Jianfeng Gao.
](https://arxiv.org/abs/2107.00652), by Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, Baining Guo.


### GAN ###
1. **[TransGAN](./gan/transGAN)** (from Seoul National University and NUUA), released with paper [TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up](https://arxiv.org/abs/2102.07074), by Yifan Jiang, Shiyu Chang, Zhangyang Wang.
2. **[Styleformer](./gan/Styleformer)** (from Facebook and Sorbonne), released with paper [Styleformer: Transformer based Generative Adversarial Networks with Style Vector](https://arxiv.org/abs/2106.07023), by Jeeseung Park, Younggeun Kim.
#### 即将更新: ####
1. **[ViTGAN]()** (from UCSD/Google), released with paper [ViTGAN: Training GANs with Vision Transformers](https://arxiv.org/pdf/2107.04589), by Kwonjoon Lee, Huiwen Chang, Lu Jiang, Han Zhang, Zhuowen Tu, Ce Liu.



## 安装
### 准备
* Linux/MacOS/Windows
* Python 3.6/3.7
* PaddlePaddle 2.1.0+
* CUDA10.2+
> 注意: 建议安装最新版本的 PaddlePaddle 以避免训练PaddleViT时出现一些 CUDA 错误。  PaddlePaddle稳定版安装请参考[链接](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) ， PaddlePaddle开发版安装请参考[链接](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html#gpu). 
### 安装
1. 创建Conda虚拟环境并激活.
   ```shell
   conda create -n paddlevit python=3.7 -y
   conda activate paddlevit
   ```
2. 按照官方说明安装 PaddlePaddle, e.g.,
   ```shell
   conda install paddlepaddle-gpu==2.1.2 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
   ```
   > 注意: 请根据您的环境更改 paddlepaddle 版本 和 cuda 版本.

3. 安装依赖项.
    *  通用的依赖项:
        ```
        pip install yacs pyyaml
        ```
    *  分割需要的依赖项:
        ```
        pip install cityscapesScripts
        ```
        安装 `detail` package:
        ```shell
        git clone https://github.com/ccvl/detail-api
        cd detail-api/PythonAPI
        make
        make install
        ```
    *  GAN需要的依赖项:
        ```
        pip install lmdb
        ```
4. 从GitHub克隆项目
    ```
    git clone https://github.com/BR-IDL/PaddleViT.git 
    ```


## 预训练模型和下载 (Model Zoo) ## 
### 图像分类 ###
| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop pct | Interp | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| vit_base_patch32_224          | 80.68 | 95.61 | 88.2M   | 4.4G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1DPEhEuu9sDdcmOPukQbR7ZcHq2bxx9cr/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ppOLj5SWlJmA-NjoLCoYIw)(ubyr) |
| vit_base_patch32_384          | 83.35 | 96.84 | 88.2M   | 12.7G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1nCOSwrDiFBFmTkLEThYwjL9SfyzkKoaf/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1jxnL00ocpmdiPM4fOu4lpg)(3c2f) |
| vit_base_patch16_224          | 84.58 | 97.30 | 86.4M   | 17.0G  | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/13D9FqU4ISsGxWXURgKW9eLOBV-pYPr-L/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ms3o2fHMQpIoVqnEHitRtA)(qv4n) |
| vit_base_patch16_384          | 85.99 | 98.00 | 86.4M   | 49.8G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1kWKaAgneDx0QsECxtf7EnUdUZej6vSFT/view?usp=sharing)/[baidu](https://pan.baidu.com/s/15ggLdiL98RPcz__SXorrXA)(wsum) |
| vit_large_patch16_224         | 85.81 | 97.82 | 304.1M  | 59.9G  | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1jgwtmtp_cDWEhZE-FuWhs7lCdpqhAMft/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1HRxUJAwEiKgrWnJSjHyU0A)(1bgk) |
| vit_large_patch16_384         | 87.08 | 98.30 | 304.1M  | 175.9G | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1zfw5mdiIm-mPxxQddBFxt0xX-IR-PF2U/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1KvxfIpMeitgXAUZGr5HV8A)(5t91) |
| vit_large_patch32_384         | 81.51 | 96.09 | 306.5M  | 44.4G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1Py1EX3E35jL7DComW-29Usg9788BB26j/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1W8sUs0pObOGpohP4vsT05w)(ieg3) |
| | | | | | | | | |
| swin_t_224   					| 81.37 | 95.54 | 28.3M   | 4.4G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1v_wzWv3TaQ0RKkKwRQwuDPzwpOb_jGEs/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1tbc751RVh3fIRsrLzrmeOw)(h2ac) |
| swin_s_224   					| 83.21 | 96.32 | 49.6M   | 8.6G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1lrODzr8zIOU9sBrH2x3zolMOS4mv4o7x/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1rlXL0tjLWbWnkIt_2Ne8Jw)(ydyx) |
| swin_b_224   					| 83.60 | 96.46 | 87.7M   | 15.3G  | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1hjEVODThNEDAlIqkg8C1KzUh3KsVNu6R/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ucSHBiuiG2sHAmR1N1JENQ)(h4y6) |
| swin_b_384   					| 84.48 | 96.89 | 87.7M   | 45.5G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1szLgwhB6WJu02Me6Uyz94egk8SqKlNsd/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1t0oXbqKNwpUAMJV7VTzcNw)(7nym) |
| swin_b_224_22kto1k    		| 85.27 | 97.56 | 87.7M   | 15.3G  | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1FhdlheMUlJzrZ7EQobpGRxd3jt3aQniU/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1KBocL_M6YNW1ZsK-GYFiNw)(6ur8) |
| swin_b_384_22kto1k    		| 86.43 | 98.07 | 87.7M   | 45.5G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1zVwIrJmtuBSiSVQhUeblRQzCKx-yWNCA/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1NziwdsEJtmjfGCeUFgtZXA)(9squ) |
| swin_l_224_22kto1k    		| 86.32 | 97.90 | 196.4M  | 34.3G  | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1yo7rkxKbQ4izy2pY5oQ5QAnkyv7zKcch/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1GsUJbSkGxlGsBYsayyKjVg)(nd2f) |
| swin_l_384_22kto1k    		| 87.14 | 98.23 | 196.4M  | 100.9G | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1-6DEvkb-FMz72MyKtq9vSPKYBqINxoKK/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1JLdS0aTl3I37oDzGKLFSqA)(5g5e) |
| | | | | | | | | |
| deit_tiny_distilled_224   	| 74.52 | 91.90 | 5.9M    | 1.1G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1fku9-11O_gQI7UpZTjagVeND-pcHbV0C/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1hAQ_85wWkqQ7sIGO1CmO9g)(rhda) |
| deit_small_distilled_224  	| 81.17 | 95.41 | 22.4M   | 4.3G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1RIeWTdf5o6pwkjqN4NbW91GZSOCalI5t/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1wCVrukvwxISAGGjorPw3iw)(pv28) |
| deit_base_distilled_224  		| 83.32 | 96.49 | 87.2M   | 17.0G  | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/12_x6-NN3Jde2BFUih4OM9NlTwe9-Xlkw/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ZnmAWgT6ewe7Vl3Xw_csuA)(5f2g) |
| deit_base_distilled_384  		| 85.43 | 97.33 | 87.2M   | 49.9G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1i5H_zjSdHfM-Znv89DHTv9ChykWrIt8I/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1PQsQIci4VCHY7l2tCzMklg)(qgj2) |
| | | | | | | | | |
| volo_d1_224  					| 84.12 | 96.78 | 26.6M   | 6.6G   | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1kNNtTh7MUWJpFSDe_7IoYTOpsZk5QSR9/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1EKlKl2oHi_24eaiES67Bgw)(xaim) |
| volo_d1_384  					| 85.24 | 97.21 | 26.6M   | 19.5G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1fku9-11O_gQI7UpZTjagVeND-pcHbV0C/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1qZWoFA7J89i2aujPItEdDQ)(rr7p) |
| volo_d2_224  					| 85.11 | 97.19 | 58.6M   | 13.7G  | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1KjKzGpyPKq6ekmeEwttHlvOnQXqHK1we/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1JCK0iaYtiOZA6kn7e0wzUQ)(d82f) |
| volo_d2_384  					| 86.04 | 97.57 | 58.6M   | 40.7G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1uLLbvwNK8N0y6Wrq_Bo8vyBGSVhehVmq/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1e7H5aa6miGpCTCgpK0rm0w)(9cf3) |
| volo_d3_224  					| 85.41 | 97.26 | 86.2M   | 19.8G  | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1OtOX7C29fJ20ESKQnYGevp4euxhmXKAT/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1vhARtV2wfI6EFf0Ap71xwg)(a5a4) |
| volo_d3_448  					| 86.50 | 97.71 | 86.2M   | 80.3G  | 448        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1lHlYhra1NNp0dp4NWaQ9SMNNmw-AxBNZ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Q6KiQw4Vu1GPm5RF9_eycg)(uudu) |
| volo_d4_224  					| 85.89 | 97.54 | 192.8M  | 42.9G  | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/16oXN7xuy-mkpfeD-loIVOK95PfptHhpX/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1PE83ZLd5evkKmHJ1V2KDsg)(vcf2) |
| volo_d4_448  					| 86.70 | 97.85 | 192.8M  | 172.5G | 448        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1N9-1OhPewA5TBR9CX5oA10obDS8e4Cfa/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1QoJ2Sqe1SK9hxbmV4uZiyg)(nd4n) |
| volo_d5_224  					| 86.08 | 97.58 | 295.3M  | 70.6G  | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1fcrvOGbAmKUhqJT-pU3MVJZQJIe4Qina/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1nqDcXMW00v9PKr3RQI-g1w)(ymdg) |
| volo_d5_448  					| 86.92 | 97.88 | 295.3M  | 283.8G | 448        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1aFXEkpfLhmQlDQHUYCuFL8SobhxUzrZX/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1K4FBv6fnyMGcAXhyyybhgw)(qfcc) |
| volo_d5_512  					| 87.05 | 97.97 | 295.3M  | 371.3G | 512        | 1.15     | bicubic       | [google](https://drive.google.com/file/d/1CS4-nv2c9FqOjMz7gdW5i9pguI79S6zk/view?usp=sharing)/[baidu](https://pan.baidu.com/s/16Wseyiqvv0MQJV8wwFDfSA)(353h) |
| | | | | | | | | |
| cswin_tiny_224  				| 82.81 | 96.30 | 22.3M   | 4.2G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1l-JY0u7NGyD6SjkyiyNnDx3wFFT1nAYO/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1L5FqU7ImWAhQHAlSilqVAw)(4q3h) |
| cswin_small_224 				| 83.60 | 96.58 | 34.6M   | 6.5G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/10eEBk3wvJdQ8Dy58LvQ11Wk1K2UfPy-E/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1FiaNiWyAuWu1IBsUFLUaAw)(gt1a) |
| cswin_base_224  				| 84.23 | 96.91 | 77.4M   | 14.6G  | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1YufKh3DKol4-HrF-I22uiorXSZDIXJmZ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1koy8hXyGwvgAfUxdlkWofg)(wj8p) |
| cswin_base_384  				| 85.51 | 97.48 | 77.4M   | 43.1G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1qCaFItzFoTYBo-4UbGzL6M5qVDGmJt4y/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1WNkY7o_vP9KJ8cd5c7n2sQ)(rkf5) |
| cswin_large_224 				| 86.52 | 97.99 | 173.3M  | 32.5G  | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1V1hteGK27t1nI84Ac7jdWfydBLLo7Fxt/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1KgIX6btML6kPiPGkIzvyVA)(b5fs) |
| cswin_large_384 				| 87.49 | 98.35 | 173.3M  | 96.1G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1LRN_6qUz71yP-OAOpN4Lscb8fkUytMic/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1eCIpegPj1HIbJccPMaAsew)(6235) |
| | | | | | | | | |
| cait_xxs24_224                | 78.38 | 94.32 | 11.9M   | 2.2G   | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1LKsQUr824oY4E42QeUEaFt41I8xHNseR/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1YIaBLopKIK5_p7NlgWHpGA)(j9m8) |
| cait_xxs36_224                | 79.75 | 94.88 | 17.2M   | 33.1G  | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1zZx4aQJPJElEjN5yejUNsocPsgnd_3tS/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1pdyFreRRXUn0yPel00-62Q)(nebg) |
| cait_xxs24_384                | 80.97 | 95.64 | 11.9M   | 6.8G   | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1J27ipknh_kwqYwR0qOqE9Pj3_bTcTx95/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1uYSDzROqCVT7UdShRiiDYg)(2j95) |
| cait_xxs36_384                | 82.20 | 96.15 | 17.2M   | 10.1G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/13IvgI3QrJDixZouvvLWVkPY0J6j0VYwL/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1GafA8B6T3h_vtmNNq2HYKg)(wx5d) |
| cait_s24_224                  | 83.45 | 96.57 | 46.8M   | 8.7G   | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1sdCxEw328yfPJArf6Zwrvok-91gh7PhS/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1BPsAMEcrjtnbOnVDQwZJYw)(m4pn) |
| cait_xs24_384                 | 84.06 | 96.89 | 26.5M   | 15.1G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1zKL6cZwqmvuRMci-17FlKk-lA-W4RVte/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1w10DPJvK8EwhOCm-tZUpww)(scsv) |
| cait_s24_384                  | 85.05 | 97.34 | 46.8M   | 26.5G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1klqBDhJDgw28omaOpgzInMmfeuDa7NAi/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1-aNO6c7Ipm9x1hJY6N6G2g)(dnp7) |
| cait_s36_384                  | 85.45 | 97.48 | 68.1M   | 39.5G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1m-55HryznHbiUxG38J2rAa01BYcjxsRZ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1-uWg-JHLEKeMukFFctoufg)(e3ui) |
| cait_m36_384                  | 86.06 | 97.73 | 270.7M  | 156.2G | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1WJjaGiONX80KBHB3YN8mNeusPs3uDhR2/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1aZ9bEU5AycmmfmHAqZIaLA)(r4hu) |
| cait_m48_448                  | 86.49 | 97.75 | 355.8M  | 287.3G | 448        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1lJSP__dVERBNFnp7im-1xM3s_lqEe82-/view?usp=sharing)/[baidu](https://pan.baidu.com/s/179MA3MkG2qxFle0K944Gkg)(imk5) |
| | | | | | | | | |
| pvtv2_b0 						| 70.47	| 90.16	| 3.7M    | 0.6G   | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/1wkx4un6y7V87Rp_ZlD4_pV63QRst-1AE/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1mab4dOtBB-HsdzFJYrvgjA)(dxgb) |
| pvtv2_b1 						| 78.70	| 94.49	| 14.0M   | 2.1G   | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/11hqLxL2MTSnKPb-gp2eMZLAzT6q2UsmG/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Ur0s4SEOxVqggmgq6AM-sQ)(2e5m) |
| pvtv2_b2 						| 82.02	| 95.99	| 25.4M   | 4.0G   | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/1-KY6NbS3Y3gCaPaUam0v_Xlk1fT-N1Mz/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1FWx0QB7_8_ikrPIOlL7ung)(are2) |
| pvtv2_b2_linear 				| 82.06	| 96.04	| 22.6M   | 3.9G   | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/1hC8wE_XanMPi0_y9apEBKzNc4acZW5Uy/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1IAhiiaJPe-Lg1Qjxp2p30w)(a4c8) |
| pvtv2_b3 						| 83.14	| 96.47	| 45.2M   | 6.8G   | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/16yYV8x7aKssGYmdE-YP99GMg4NKGR5j1/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ge0rBsCqIcpIjrVxsrFhnw)(nc21) |
| pvtv2_b4 						| 83.61	| 96.69	| 62.6M   | 10.0G  | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/1gvPdvDeq0VchOUuriTnnGUKh0N2lj-fA/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1VMSD_Kr_hduCZ5dxmDbLoA)(tthf) |
| pvtv2_b5 						| 83.77	| 96.61	| 82.0M   | 11.5G  | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/1OHaHiHN_AjsGYBN2gxFcQCDhBbTvZ02g/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ey4agxI2Nb0F6iaaX3zAbA)(9v6n) |
| | | | | | | | | | 
| shuffle_vit_tiny  			| 82.39 | 96.05 | 28.5M   | 4.6G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1ffJ-tG_CGVXztPEPQMaT_lUoc4hxFy__/view?usp=sharing)/[baidu](https://pan.baidu.com/s/19DhlLIFyPGOWtyq_c83ZGQ)(8a1i) |
| shuffle_vit_small 			| 83.53 | 96.57 | 50.1M   | 8.8G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1du9H0SKr0QH9GQjhWDOXOnhpSVpfbb8X/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1rM2J8BVwxQ3kRZoHngwNZA)(xwh3) |
| shuffle_vit_base  			| 83.95 | 96.91 | 88.4M   | 15.5G  | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1sYh808AyTG3-_qv6nfN6gCmyagsNAE6q/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1fks_IYDdnXdAkCFuYHW_Nw)(1gsr) |
| | | | | | | | | |
| t2t_vit_7      				| 71.68 | 90.89 | 4.3M    | 1.0G   | 224   	    | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1YkuPs1ku7B_udydOf_ls1LQvpJDg_c_j/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1jVNsz37gatLCDaOoU3NaMA)(1hpa) |
| t2t_vit_10     				| 75.15 | 92.80 | 5.8M    | 1.3G   | 224   	    | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1H--55RxliMDlOCekn7FpKrHDGsUkyrJZ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1nbdb4PFMq4nsIp8HrNxLQg)(ixug) |
| t2t_vit_12     				| 76.48 | 93.49 | 6.9M    | 1.5G   | 224   	    | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1stnIwOwaescaEcztaF1QjI4NK4jaqN7P/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1DcMzq9WeSwrS3epv6jKJXw)(qpbb) |
| t2t_vit_14     				| 81.50 | 95.67 | 21.5M   | 4.4G   | 224   	    | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1HSvN3Csgsy7SJbxJYbkzjUx9guftkfZ1/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1wcfh22uopBv7pS7rKcH_iw)(c2u8) |
| t2t_vit_19     				| 81.93 | 95.74 | 39.1M   | 7.8G   | 224   	    | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1eFnhaL6I33pHCQw2BaEE0Oet9CnjmUf_/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Hpyc5hBYo1zqoXWpryegnw)(4in3) |
| t2t_vit_24     				| 82.28 | 95.89 | 64.0M   | 12.8G  | 224   	    | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1Z7nZCHeFp0AhIkGYcMAFkKdkGN0yXtpv/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Hpyc5hBYo1zqoXWpryegnw)(4in3) |
| t2t_vit_t_14   				| 81.69 | 95.85 | 21.5M   | 4.4G   | 224   	    | 0.9      | bicubic       | [google](https://drive.google.com/file/d/16li4voStt_B8eWDXqJt7s20OT_Z8L263/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Hpyc5hBYo1zqoXWpryegnw)(4in3) |
| t2t_vit_t_19   				| 82.44 | 96.08 | 39.1M   | 7.9G   | 224   	    | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1Ty-42SYOu15Nk8Uo6VRTJ7J0JV_6t7zJ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1YdQd6l8tj5xMCWvcHWm7sg)(mier) |
| t2t_vit_t_24   				| 82.55 | 96.07 | 64.0M   | 12.9G  | 224   	    | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1cvvXrGr2buB8Np2WlVL7n_F1_CnI1qow/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1BMU3KX_TRmPxQ1jN5cmWhg)(6vxc) |
| t2t_vit_14_384 				| 83.34 | 96.50 | 21.5M   | 13.0G  | 384   	    | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1Yuso8WD7Q8Lu_9I8dTvAvkcXXtPSkmnm/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1AOMhyVRF9zPqJe-lTrd7pw)(r685) |
| | | | | | | | | |
| cross_vit_tiny_224 			| 73.20 | 91.90 | 6.9M    | 1.3G   | 224   	    | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1ILTVwQtetcb_hdRjki2ZbR26p-8j5LUp/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1byeUsM34_gFL0jVr5P5GAw)(scvb) |
| cross_vit_small_224 			| 81.01 | 95.33 | 26.7M   | 5.2G   | 224   	    | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1ViOJiwbOxTbk1V2Go7PlCbDbWPbjWPJH/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1I9CrpdPU_D5LniqIVBoIPQ)(32us) |
| cross_vit_base_224 			| 82.12 | 95.87 | 104.7M  | 20.2G  | 224   	    | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1vTorkc63O4JE9cYUMHBRxFMDOFoC-iK7/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1TR_aBHQ2n1J0RgHFoVh_bw)(jj2q) |
| cross_vit_9_224 				| 73.78 | 91.93 | 8.5M    | 1.6G   | 224   	    | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1UCX9_mJSx2kDAmEd_xDXyd4e6-Mg3RPf/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1M8r5vqMHJ-rFwBoW1uL2qQ)(mjcb) |
| cross_vit_15_224 				| 81.51 | 95.72 | 27.4M   | 5.2G   | 224   	    | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1HwkLWdz6A3Nz-dVbw4ZUcCkxUbPXgHwM/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1wiO_Gjk4fvSq08Ud8xKwVw)(n55b) |
| cross_vit_18_224 				| 82.29 | 96.00 | 43.1M   | 8.3G   | 224   	    | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1C4b_a_6ia8NCEXSUEMDdCEFzedr0RB_m/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1w7VJ7DNqq6APuY7PdlKEjA)(xese) |
| cross_vit_9_dagger_224 		| 76.92 | 93.61 | 8.7M    | 1.7G   | 224   	    | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1_cXQ0M8Hr9UyugZk07DrsBl8dwwCA6br/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1F1tRSaG4EfCV_WiTEwXxBw)(58ah) |
| cross_vit_15_dagger_224 		| 82.23 | 95.93 | 28.1M   | 5.6G   | 224   	    | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1cCgBoozh2WFtSz42LwEUUPPyC5KmkAFg/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1xJ4P2zy3r9RcNFSMtzvZgg)(qwup) |
| cross_vit_18_dagger_224 		| 82.51 | 96.03 | 44.1M   | 8.7G   | 224   	    | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1sdAbWxKL5k3QIo1zdgHzasIOtpy_Ogpw/view?usp=sharing)/[baidu](https://pan.baidu.com/s/15qYHgt0iRxdhtXoC_ct2Jg)(qtw4) |
| cross_vit_15_dagger_384 		| 83.75 | 96.75 | 28.1M   | 16.4G  | 384   	    | 1.0      | bicubic       | [google](https://drive.google.com/file/d/12LQjYbs9-LyrY1YeRt46x9BTB3NJuhpJ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1d-BAm03azLP_CyEHF3c7ZQ)(w71e) |
| cross_vit_18_dagger_384 		| 84.17 | 96.82 | 44.1M   | 25.8G  | 384   	    | 1.0 	   | bicubic       | [google](https://drive.google.com/file/d/1CeGwB6Tv0oL8QtL0d7Ar-d02Lg_PqACr/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1l_6PTldZ3IDB7XWgjM6LhA)(99b6) |
| | | | | | | | | | 
| beit_base_patch16_224_pt22k   | 85.21 | 97.66 | 87M    | 12.7G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1lq5NeQRDHkIQi7U61OidaLhNsXTWfh_Z/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1pjblqaESqfXVrpgo58oR6Q)(fshn) |
| beit_base_patch16_384_pt22k   | 86.81 | 98.14 | 87M    | 37.3G   | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1wn2NS7kUdlERkzWEDeyZKmcRbmWL7TR2/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1WVbNjxuIUh514pKAgZZEzg)(arvc) |
| beit_large_patch16_224_pt22k  | 87.48 | 98.30 | 304M   | 45.0G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/11OR1FKxzfafqT7GzTW225nIQjxmGSbCm/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1bvhERVXN2TyRcRJFzg7sIA)(2ya2) |
| beit_large_patch16_384_pt22k  | 88.40 | 98.60 | 304M   | 131.7G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/10EraafYS8CRpEshxClOmE2S1eFCULF1Y/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1H76G2CGLY3YmmYt4-suoRA)(qtrn) |
| beit_large_patch16_512_pt22k  | 88.60 | 98.66 | 304M   | 234.0G  | 512        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1xIIocftsB1PcDHZttPqLdrJ-G4Tyfrs-/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1WtTVK_Wvg-izaF0M6Gzw-Q)(567v) |
| | | | | | | | | | 
| Focal-T    					| 82.03 | 95.86 | 28.9M   | 4.9G    | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1HzZJbYH_eIo94h0wLUhqTyJ6AYthNKRh/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1JCr2qIA-SZvTqbTO-m2OwA)(i8c2) |
| Focal-T (use conv)   			| 82.70 | 96.14 | 30.8M   | 4.9G    | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1PS0-gdXHGl95LqH5k5DG62AH6D3i7v0D/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1tVztox4bVJuJEjkD1fLaHQ)(smrk) |
| Focal-S    					| 83.55 | 96.29 | 51.1M   | 9.4G    | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1HnVAYsI_hmiomyS4Ax3ccPE7gk4mlTU8/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1b7uugAY9RhrgTkUwYcvvow)(dwd8) |
| Focal-S (use conv)   			| 83.85 | 96.47 | 53.1M   | 9.4G    | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1vcHjYiGNMayoSTPoM8z39XRH6h89TB9V/view?usp=sharing)/[baidu](https://pan.baidu.com/s/174a2aZzCEt3teLuAnIzMtA)(nr7n) |
| Focal-B    					| 83.98 | 96.48 | 89.8M   | 16.4G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1bNMegxetWpwZNcmDEC3MHCal6SNXSgWR/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1piBslNhxWR78aQJIdoZjEw)(8akn) |
| Focal-B (use conv)   			| 84.18 | 96.61 | 93.3M   | 16.4G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1-J2gDnKrvZGtasvsAYozrbMXR2LtIJ43/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1GTLfnTlt6I6drPdfSWB1Iw)(5nfi) |
| | | | | | | | | | 
| mobilevit_xxs   				| 70.31| 89.68 | 1.32M   | 0.44G   | 256        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1l3L-_TxS3QisRUIb8ohcv318vrnrHnWA/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1KFZ5G834_-XXN33W67k8eg)(axpc) |
| mobilevit_xs   				| 74.47| 92.02 | 2.33M   | 0.95G   | 256        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1oRMA4pNs2Ba0LYDbPufC842tO4OFcgwq/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1IP8S-S6ZAkiL0OEsiBWNkw)(hfhm) |
| mobilevit_s   				| 76.74| 93.08 | 5.59M   | 1.88G   | 256        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1ibkhsswGYWvZwIRjwfgNA4-Oo2stKi0m/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1-rI6hiCHZaI7os2siFASNg)(34bg) |
| mobilevit_s $\dag$  			| 77.83| 93.83 | 5.59M   | 1.88G   | 256        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1BztBJ5jzmqgDWfQk-FB_ywDWqyZYu2yG/view?usp=sharing)/[baidu](https://pan.baidu.com/s/19YepMAO-sveBOLA4aSjIEQ?pwd=92ic)(92ic) |
| | | | | | | | | | 
| vip_s7  						| 81.50 | 95.76 | 25.1M   | 7.0G   |    224     | 0.875    | bicubic       | [google](https://drive.google.com/file/d/16bZkqzbnN08_o15k3MzbegK8SBwfQAHF/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1uY0FsNPYaM8cr3ZCdAoVkQ)(mh9b) |
| vip_m7  						| 82.75 | 96.05 | 55.3M   | 16.4G  |    224     | 0.875    | bicubic       | [google](https://drive.google.com/file/d/11lvT2OXW0CVGPZdF9dNjY_uaEIMYrmNu/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1j3V0Q40iSqOY15bTKlFFRw)(hvm8) |
| vip_l7  						| 83.18 | 96.37 | 87.8M   | 24.5G  |    224     | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1bK08JorLPMjYUep_TnFPKGs0e1j0UBKJ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1I5hnv3wHWEaG3vpDqaNL-w)(tjvh) |
| | | | | | | | | | 
| xcit_nano_12_p16_224_dist   | 72.32  | 90.86  | 0.6G    | 3.1M      | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/14FsYtm48JB-rQFF9CanJsJaPESniWD7q/view?usp=sharing)/[baidu](https://pan.baidu.com/s/15kdY4vzwU2QiBSU5127AYA)(7qvz)     |
| xcit_nano_12_p16_384_dist   | 75.46  | 92.70  | 1.6G    | 3.1M      | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1zR-hFQryocF9muG-erzcxFuJme5y_e9f/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1449qtQzEMg6lqdtClyiCRQ)(1y2j)     |
| xcit_large_24_p16_224_dist  | 84.92  | 97.13  | 35.9G   | 189.1M    | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1lAtko_KwOagjwaFvUkeXirVClXCV8gt-/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Gs401mXqG1bifi1hBdXtig)(kfv8)     |
| xcit_large_24_p16_384_dist  | 85.76  | 97.54  | 105.5G  | 189.1M    | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/15djnKz_-eooncvyZp_UTwOiHIm1Hxo_G/view?usp=sharing)/[baidu](https://pan.baidu.com/s/14583hbtIVbZ_2ifZepQItQ)(ffq3)     |
| xcit_nano_12_p8_224_dist    | 76.33  | 93.10  | 2.2G    | 3.0M      | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1XxRNjskLvSVp6lvhlsnylq6g7vd_5MsI/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1DZJxuahFJyz-rEEsCqhhrA)(jjs7)     |
| xcit_nano_12_p8_384_dist    | 77.82  | 94.04  | 6.3G    | 3.0M      | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1P3ln8JqLzMKbJAhCanRbu7i5NMPVFNec/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ECY9-PVDMNSup8NMQiqBrw)(dmc1)     |
| xcit_large_24_p8_224_dist   | 85.40  | 97.40  | 141.4G  | 188.9M    | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/14ZoDxEez5NKVNAsbgjTPisjOQEAA30Wy/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1D_zyvjzIVFp6iqx1s7IEbA)(y7gw)     |
| xcit_large_24_p8_384_dist   | 85.99  | 97.69  | 415.5G  | 188.9M    | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1stcUwwFNJ38mdaFsNXq24CBMmDenJ_e4/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1lwbBk7GFuqnnP_iU2OuDRw)(9xww)     |
| | | | | | | | | |
| pit_ti 	     | 72.91	| 91.40	| 4.8M    | 0.5G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1bbeqzlR_CFB8CAyTUN52p2q6ii8rt0AW/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Yrq5Q16MolPYHQsT_9P1mw)(ydmi)  |
| pit_ti_distill | 74.54	| 92.10 | 5.1M    | 0.5G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1m4L0OVI0sYh8vCv37WhqCumRSHJaizqX/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1RIM9NGq6pwfNN7GJ5WZg2w)(7k4s)  |
| pit_xs 	     | 78.18    | 94.16 | 10.5M   | 1.1G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1qoMQ-pmqLRQmvAwZurIbpvgMK8MOEgqJ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/15d7ep05vI2UoKvL09Zf_wg)(gytu)  |
| pit_xs_distill | 79.31 	| 94.36 | 10.9M   | 1.1G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1EfHOIiTJOR-nRWE5AsnJMsPCncPHEgl8/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1DqlgVF7U5qHfGD3QJAad4A)(ie7s)  |
| pit_s  		 | 81.08 	| 95.33 | 23.4M   | 2.4G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1TDSybTrwQpcFf9PgCIhGX1t-f_oak66W/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Vk-W1INskQq7J5Qs4yphCg)(kt1n)  |
| pit_s_distill  | 81.99 	| 95.79 | 24.0M   | 2.5G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1U3VPP6We1vIaX-M3sZuHmFhCQBI9g_dL/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1L7rdWmMW8tiGkduqmak9Fw)(hhyc)  |
| pit_b   		 | 82.44 	| 95.71 | 73.5M	  | 10.6G  | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1-NBZ9-83nZ52jQ4DNZAIj8Xv6oh54nx-/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1XRDPY4OxFlDfl8RMQ56rEg)(uh2v)  |
| pit_b_distill  | 84.14 	| 96.86 | 74.5M   | 10.7G  | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/12Yi4eWDQxArhgQb96RXkNWjRoCsDyNo9/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1vJOUGXPtvC0abg-jnS4Krw)(3e6g)  |
| | | | | | | | | |
| halonet26t 	 | 79.10	| 94.31	| 12.5M    | 3.2G   | 256        | 0.95     | bicubic       |[google](https://drive.google.com/file/d/1F_a1brftXXnPM39c30NYe32La9YZQ0mW/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1FSlSTuYMpwPJpi4Yz2nCTA)(ednv)  |
| halonet50ts 	 | 81.65	| 95.61	| 22.8M    | 5.1G   | 256        | 0.94     | bicubic       |[google](https://drive.google.com/file/d/12t85kJcPA377XePw6smch--ELMBo6p0Y/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1X4LM-sqoTKG7CrM5BNjcdA)(3j9e)  |
| | | | | | | | | |
| poolformer_s12 | 77.24 | 93.51 | 11.9M   | 1.8G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/15EBfTTU6coLCsDNiLgAWYiWeMpp3uYH4/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1n6TUxQGlssTu4lyLrBOXEw)(zcv4)             |
| poolformer_s24 | 80.33 | 95.05 | 21.3M   | 3.4G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1JxqJluDpp1wwe7XtpTi1aWaVvlq0Q3xF/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1d2uyHB5R6ZWPzXWhdtm6fw)(nedr)             |
| poolformer_s36 | 81.43 | 95.45 | 30.8M   | 5.0G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1ka3VeupDRFBSzzrcw4wHXKGqoKv6sB_Y/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1de6ZJkmYEmVI7zKUCMB_xw)(fvpm)             |
| poolformer_m36 | 82.11 | 95.69 | 56.1M   | 8.9G   | 224        | 0.95     | bicubic       | [google](https://drive.google.com/file/d/1LTZ8wNRb_GSrJ9H3qt5-iGiGlwa4dGAK/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1qNTYLw4vyuoH1EKDXEcSvw)(whfp)             |
| poolformer_m48 | 82.46 | 95.96 | 73.4M   | 11.8G  | 224        | 0.95     | bicubic       | [google](https://drive.google.com/file/d/1YhXEVjWtI4bZB_Qwama8G4RBanq2K15L/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1VJXANTseTUEA0E6HYf-XyA)(374f)             |
| | | | | | | | | |
| botnet50 	 | 77.38	| 93.56	| 20.9M    | 5.3G   | 224        | 0.875     | bicubic       |[google](https://drive.google.com/file/d/1S4nxgRkElT3K4lMx2JclPevmP3YUHNLw/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1CW40ShBJQYeFgdBIZZLSjg)(wh13)
| | | | | | | | | |
| CvT-13-224      | 81.59 | 95.67 | 20M    | 4.5G    | 224        | 0.875      | bicubic       | [google](https://drive.google.com/file/d/1r0fnHn1bRPmN0mi8RwAPXmD4utDyOxEf/view?usp=sharing)/[baidu](https://pan.baidu.com/s/13xNwCGpdJ5MVUi369OGl5Q)(vev9) |
| CvT-21-224      | 82.46 | 96.00 | 32M    | 7.1G    | 224        | 0.875      | bicubic       | [google](https://drive.google.com/file/d/18s7nRfvcmNdbRuEpTQe02AQE3Y9UWVQC/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1mOjbMNoQb7X3VJD3LV0Hhg)(t2rv) |
| CvT-13-384   	  | 83.00 | 96.36 | 20M    | 16.3G   | 384        | 1.0        | bicubic       | [google](https://drive.google.com/file/d/1J0YYPUsiXSqyExBPtOPrOLL9c16syllg/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1upITRr5lNHLjbBJtIr-jdg)(wswt) |
| CvT-21-384   	  | 83.27 | 96.16 | 32M    | 24.9G   | 384        | 1.0        | bicubic       | [google](https://drive.google.com/file/d/1tpXv_yYXtvyArlYi7AFcHUOqemhyMWHW/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1hXKi3Kb7mNxPFVmR6cdkMg)(hcem) |
| CvT-13-384-22k  | 83.26 | 97.09 | 20M    | 16.3G   | 384        | 1.0        | bicubic       | [google](https://drive.google.com/file/d/18djrvq422u1pGLPxNfWAp6d17F7C5lbP/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1YYv5rKPmroxKCnzkesUr0g)(c7m9) |
| CvT-21-384-22k  | 84.91 | 97.62 | 32M    | 24.9G   | 384        | 1.0        | bicubic       | [google](https://drive.google.com/file/d/1NVXd7vxVoRpL-21GN7nGn0-Ut0L0Owp8/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1N3xNU6XFHb1CdEOrnjKuoA)(9jxe) |
| CvT-w24-384-22k | 87.58 | 98.47 | 277M   | 193.2G  | 384        | 1.0        | bicubic       | [google](https://drive.google.com/file/d/1M3bg46N4SGtupK8FcvAOE0jltOwP5yja/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1MNJurm8juHRGG9SAw3IOkg)(bbj2) |
| | | | | | | | | |
| HVT-Ti-1       | 69.45 | 89.28 | 5.7M    | 0.6G   | 224        |  0.875   |  bicubic      |  [google](https://drive.google.com/file/d/11BW-qLBMu_1TDAavlrAbfVlXB53dgm42/view?usp=sharing)/[baidu](https://pan.baidu.com/s/16rZvJqL-UVuWFsCDuxFDqg?pwd=egds)(egds) |
| HVT-S-0        | 80.30 | 95.15 | 22.0M   | 4.6G   | 224        |  0.875   |  bicubic      |  [google](https://drive.google.com/file/d/1GlJ2j2QVFye1tAQoUJlgKTR_KELq3mSa/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1L-tjDxkQx00jg7BsDClabA?pwd=hj7a)(hj7a) |
| HVT-S-1        | 78.06 | 93.84 | 22.1M   | 2.4G   | 224        |  0.875   |  bicubic      |  [google](https://drive.google.com/file/d/16H33zNIpNrHBP1YhCq4zmLjRYQJ0XEmX/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1quOsgVuxTcauISQ3SehysQ?pwd=tva8)(tva8) |
| HVT-S-2        | 77.41 | 93.48 | 22.1M   | 1.9G   | 224        |  0.875   |  bicubic      |  [google](https://drive.google.com/file/d/1U14LA7SXJtFep_SdUCjAV-cDOQ9A_OFk/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1nooWTBzaXyBtEgadn9VDmw?pwd=bajp)(bajp) |
| HVT-S-3        | 76.30 | 92.88 | 22.1M   | 1.6G   | 224        |  0.875   |  bicubic      |  [google](https://drive.google.com/file/d/1m1CjOcZfPMLDRyX4QBgMhHV1m6rtu44v/view?usp=sharing)/[baidu](https://pan.baidu.com/s/15sAOmQN6Hx0GLelYDuMQXw?pwd=rjch)(rjch) |
| HVT-S-4        | 75.21 | 92.34 | 22.1M   | 1.6G   | 224        |  0.875   |  bicubic      |  [google](https://drive.google.com/file/d/14comGo9lO12dUeGGL52MuIJWZPSit7I0/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1o31hMRWR7FTCjUk7_fAOgA?pwd=ki4j)(ki4j) |
| | | | | | | | | |
| | | | | | | | | |
| mlp_mixer_b16_224            	| 76.60 | 92.23 | 60.0M   | 12.7G  | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1ZcQEH92sEPvYuDc6eYZgssK5UjYomzUD/view?usp=sharing)/[baidu](https://pan.baidu.com/s/12nZaWGMOXwrCMOIBfUuUMA)(xh8x) |
| mlp_mixer_l16_224           	| 72.06 | 87.67 | 208.2M  | 44.9G  | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1mkmvqo5K7JuvqGm92a-AdycXIcsv1rdg/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1AmSVpwCaGR9Vjsj_boL7GA)(8q7r) |
| | | | | | | | | |
| resmlp_24_224                	| 79.38 | 94.55 | 30.0M   | 6.0G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/15A5q1XSXBz-y1AcXhy_XaDymLLj2s2Tn/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1nLAvyG53REdwYNCLmp4yBA)(jdcx) |
| resmlp_36_224             	| 79.77 | 94.89 | 44.7M   | 9.0G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1WrhVm-7EKnLmPU18Xm0C7uIqrg-RwqZL/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1QD4EWmM9b2u1r8LsnV6rUA)(33w3) |
| resmlp_big_24_224         	| 81.04 | 95.02 | 129.1M  | 100.7G | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1KLlFuzYb17tC5Mmue3dfyr2L_q4xHTZi/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1oXU6CR0z7O0XNwu_UdZv_w)(r9kb) |
| resmlp_12_distilled_224 		| 77.95 | 93.56 | 15.3M   |	3.0G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1cDMpAtCB0pPv6F-VUwvgwAaYtmP8IfRw/view?usp=sharing)/[baidu](https://pan.baidu.com/s/15kJeZ_V1MMjTX9f1DBCgnw)(ghyp) |
| resmlp_24_distilled_224 		| 80.76 | 95.22 | 30.0M   |	6.0G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/15d892ExqR1sIAjEn-cWGlljX54C3vihA/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1NgQtSwuAwsVVOB8U6N4Aqw)(sxnx) |
| resmlp_36_distilled_224 		| 81.15 | 95.48 | 44.7M	  | 9.0G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1Laqz1oDg-kPh6eb6bekQqnE0m-JXeiep/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1p1xGOJbMzH_RWEj36ruQiw)(vt85) |
| resmlp_big_24_distilled_224 	| 83.59 | 96.65 | 129.1M  |	100.7G | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/199q0MN_BlQh9-HbB28RdxHj1ApMTHow-/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1yUrfbqW8vLODDiRV5WWkhQ)(4jk5) |
| resmlp_big_24_22k_224   		| 84.40 | 97.11 | 129.1M  | 100.7G | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1zATKq1ruAI_kX49iqJOl-qomjm9il1LC/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1VrnRMbzzZBmLiR45YwICmA)(ve7i) |
| | | | | | | | | |
| gmlp_s16_224                 	| 79.64 | 94.63 | 19.4M   | 4.5G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1TLypFly7aW0oXzEHfeDSz2Va4RHPRqe5/view?usp=sharing)/[baidu](https://pan.baidu.com/s/13UUz1eGIKyqyhtwedKLUMA)(bcth) |
| | | | | | | | | |
| ff_only_tiny (linear_tiny) 	| 61.28 | 84.06 |         |        | 224   	    | 0.875    | bicubic       | [google](https://drive.google.com/file/d/14bPRCwuY_nT852fBZxb9wzXzbPWNfbCG/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1nNE4Hh1Nrzl7FEiyaZutDA)(mjgd) |
| ff_only_base (linear_base) 	| 74.82 | 91.71 |         |        | 224   	    | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1DHUg4oCi41ELazPCvYxCFeShPXE4wU3p/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1l-h6Cq4B8kZRvHKDTzhhUg)(m1jc) |
| | | | | | | | | |
| repmlp_res50_light_224 		| 77.01 | 93.46 | 87.1M   | 3.3G   | 224   	    | 0.875    | bicubic       | [google](https://drive.google.com/file/d/16bCFa-nc_-tPVol-UCczrrDO_bCFf2uM/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1bzmpS6qJJTsOq3SQE7IOyg)(b4fg) |
| | | | | | | | | |
| cyclemlp_b1 					 | 78.85 | 94.60 | 15.1M   |    | 224   	    | 0.9    | bicubic       | [google](https://drive.google.com/file/d/10WQenRy9lfOJF4xEHc9Mekp4zHRh0mJ_/view?usp=sharing)/[baidu](https://pan.baidu.com/s/11UQp1RkWBsZFOqit_uU80w)(mnbr) |
| cyclemlp_b2 					 | 81.58 | 95.81 | 26.8M   |    | 224   	    | 0.9    | bicubic       | [google](https://drive.google.com/file/d/1dtQHCwtxNh9jgiHivN5iYpHe7uKRUjhk/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Js-Oq5vyiB7oPagn43cn3Q)(jwj9) |
| cyclemlp_b3 					 | 82.42 | 96.07 | 38.3M   |    | 224   	    | 0.9    | bicubic       | [google](https://drive.google.com/file/d/11kMq112tAwVE5llJIepIIixz74AjaJhU/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1b7cau1yPxqATA8X7t2DXkw)(v2fy) |
| cyclemlp_b4 					 | 82.96 | 96.33 | 51.8M   |    | 224   	    | 0.875  | bicubic       | [google](https://drive.google.com/file/d/1vwJ0eD9Ic-NvLvCz1zEAmn7RxBMtd_v2/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1P3TlnXRFGWj9nVP5xBGGWQ)(fnqd) |
| cyclemlp_b5 					 | 83.25 | 96.44 | 75.7M   |    | 224   	    | 0.875  | bicubic       | [google](https://drive.google.com/file/d/12_I4cfOBfp7kC0RvmnMXFqrSxww6plRW/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1-Cka1tNqGUQutkAP3VZXzQ)(s55c) |
| | | | | | | | | |
| convmixer_1024_20  			| 76.94 | 93.35 | 24.5M   | 9.5G   |    224     | 0.96     | bicubic       | [google](https://drive.google.com/file/d/1R7zUSl6_6NFFdNOe8tTfoR9VYQtGfD7F/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1DgGA3qYu4deH4woAkvjaBw)(qpn9) |
| convmixer_768_32  			| 80.16 | 95.08 | 21.2M   | 20.8G  |    224     | 0.96     | bicubic       | [google](https://drive.google.com/file/d/196Lg_Eet-hRj733BYASj22g51wdyaW2a/view?usp=sharing)/[baidu](https://pan.baidu.com/s/17CbRNzY2Sy_Cu7cxNAkWmQ)(m5s5) |
| convmixer_1536_20  			| 81.37 | 95.62 | 51.8M   | 72.4G  |    224     | 0.96     | bicubic       | [google](https://drive.google.com/file/d/1-LlAlADiu0SXDQmE34GN2GBhqI-RYRqO/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1R-gSzhzQNfkuZVxsaE4vEw)(xqty) |
| | | | | | | | | |
| convmlp_s			  			| 76.76 | 93.40 | 9.0M    | 2.4G   |    224     | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1D8kWVfQxOyyktqDixaZoGXB3wVspzjlc/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1WseHYALFB4Of3Dajmlt45g)(3jz3) |
| convmlp_m			  			| 79.03 | 94.53 | 17.4M   | 4.0G   |    224     | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1TqVlKHq-WRdT9KDoUpW3vNJTIRZvix_m/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1koipCAffG6REUyLYk0rGAQ)(vyp1) |
| convmlp_l			  			| 80.15 | 95.00 | 42.7M   | 10.0G  |    224     | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1KXxYogDh6lD3QGRtFBoX5agfz81RDN3l/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1f1aEeVoySzImI89gkjcaOA)(ne5x) |
| | | | | | | | | |















### 目标检测 ###
| Model | backbone  | box_mAP | Model                                                                                                                                                       |
|-------|-----------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DETR  | ResNet50  | 42.0    | [google](https://drive.google.com/file/d/1ruIKCqfh_MMqzq_F4L2Bv-femDMjS_ix/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1J6lB1mezd6_eVW3jnmohZA)(n5gk) |
| DETR  | ResNet101 | 43.5    | [google](https://drive.google.com/file/d/11HCyDJKZLX33_fRGp4bCg1I14vrIKYW5/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1_msuuAwFMNbAlMpgUq89Og)(bxz2) |
| Mask R-CNN | Swin-T 1x |  43.7   | [google](https://drive.google.com/file/d/1OpbCH5HuIlxwakNz4PzrAlJF3CxkLSYp/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18HALSo2RHMBsX-Gbsi-YOw)(qev7) |
| Mask R-CNN | Swin-T 3x |  46.0   | [google](https://drive.google.com/file/d/1oREwIk1ORhSsJcs4Y-Cfd0XrSEfPFP3-/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1tw607oogDWQ7Iz91ItfuGQ)(m8fg) |
| Mask R-CNN | Swin-S 3x |  48.4   | [google](https://drive.google.com/file/d/1ZPWkz0zMzHJycHd6_s2hWDHIsW8SdZcK/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ubC5_CKSq0ExQSINohukVg)(hdw5) |
| Mask R-CNN | pvtv2_b0 		|  38.3   | [google](https://drive.google.com/file/d/1wA324LkFtGezHJovSZ4luVqSxVt9woFc/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1q67ZIDSHn9Y-HU_WoQr8OQ)(3kqb) |
| Mask R-CNN | pvtv2_b1 		|  41.8   | [google](https://drive.google.com/file/d/1alNaSmR4TSXsPpGoUZr2QQf5phYQjIzN/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1aSkuDiNpxdnFWE1Wn1SWNw)(k5aq) |
| Mask R-CNN | pvtv2_b2 		|  45.2   | [google](https://drive.google.com/file/d/1tg6B5OEV4OWLsDxTCjsWgxgaSgIh4cID/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1DLwxCZVZizb5HKih7RFw2w)(jh8b) |
| Mask R-CNN | pvtv2_b2_linear 	|  44.1   | [google](https://drive.google.com/file/d/1b26vxK3QVGx5ovqKir77NyY6YPgAWAEj/view?usp=sharing)/[baidu](https://pan.baidu.com/s/16T-Nyo_Jm2yDq4aoXpdnbg)(8ipt) |
| Mask R-CNN | pvtv2_b3 		|  46.9   | [google](https://drive.google.com/file/d/1H6ZUCixCaYe1AvlBkuqYoxzz4b-icJ3u/view?usp=sharing)/[baidu](https://pan.baidu.com/s/16QVsjUOXijo5d9cO3FZ39A)(je4y) |
| Mask R-CNN | pvtv2_b4 		|  47.5   | [google](https://drive.google.com/file/d/1pXQNpn0BoKqiuVaGtJL18eWG6XmdlBOL/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1yhX7mpmb2wbRvWZFnUloBQ)(n3ay) |
| Mask R-CNN | pvtv2_b5 		|  47.4   | [google](https://drive.google.com/file/d/12vOyw6pUfK1NdOWBF758aAZuaf-rZLvx/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1-gasQk9PqLMkrWXw4aX41g)(jzq1) |

### 目标分割 ###
#### Pascal Context ####
|Model      | Backbone  | Batch_size | mIoU (ss) | mIoU (ms+flip) | Backbone_checkpoint | Model_checkpoint      |     ConfigFile  |
|-----------|-----------|------------|-----------|----------------|-----------------------------------------------|-----------------------------------------------------------------------|------------|
|SETR_Naive | ViT_large |     16     |   52.06   |      52.57        | [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)     | [google](https://drive.google.com/file/d/1AUyBLeoAcMH0P_QGer8tdeU44muTUOCA/view?usp=sharing)/[baidu](https://pan.baidu.com/s/11XgmgYG071n_9fSGUcPpDQ)(xdb8)   | [config](semantic_segmentation/configs/setr/SETR_Naive_Large_480x480_80k_pascal_context_bs_16.yaml) | 
|SETR_PUP   | ViT_large |     16     |   53.90   |       54.53    | [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)     | [google](https://drive.google.com/file/d/1IY-yBIrDPg5CigQ18-X2AX6Oq3rvWeXL/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1v6ll68fDNCuXUIJT2Cxo-A)(6sji) | [config](semantic_segmentation/configs/setr/SETR_PUP_Large_480x480_80k_pascal_context_bs_16.yaml) |
|SETR_MLA   | ViT_Large |     8      |   54.39   |       55.16       | [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)     | [google](https://drive.google.com/file/d/1utU2h0TrtuGzRX5RMGroudiDcz0z6UmV/view)/[baidu](https://pan.baidu.com/s/1Eg0eyUQXc-Mg5fg0T3RADA)(wora)| [config](semantic_segmentation/configs/setr/SETR_MLA_Large_480x480_80k_pascal_context_bs_8.yaml) |
|SETR_MLA   | ViT_large |     16     |   55.01   |       55.87        | [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)     | [google](https://drive.google.com/file/d/1SOXB7sAyysNhI8szaBqtF8ZoxSaPNvtl/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1jskpqYbazKY1CKK3iVxAYA)(76h2) | [config](semantic_segmentation/configs/setr/SETR_MLA_Large_480x480_80k_pascal_context_bs_16.yaml) |

#### Cityscapes ####
|Model      | Backbone  | Batch_size | Iteration | mIoU (ss) | mIoU (ms+flip) | Backbone_checkpoint | Model_checkpoint     |     ConfigFile  |
|-----------|-----------|------------|-----------|-----------|----------------|-----------------------------------------------|-----------------------------------------------------------------------|------------|
|SETR_Naive | ViT_Large |     8      |     40k   |   76.71   |       79.03        | [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)      | [google](https://drive.google.com/file/d/1QialLNMmvWW8oi7uAHhJZI3HSOavV4qj/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1F3IB31QVlsohqW8cRNphqw)(g7ro)  |  [config](semantic_segmentation/configs/setr/SETR_Naive_Large_768x768_40k_cityscapes_bs_8.yaml)| 
|SETR_Naive | ViT_Large |     8      |     80k   |   77.31   |       79.43      | [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)      | [google](https://drive.google.com/file/d/1RJeSGoDaOP-fM4p1_5CJxS5ku_yDXXLV/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1XbHPBfaHS56HlaMJmdJf1A)(wn6q)   |  [config](semantic_segmentation/configs/setr/SETR_Naive_Large_768x768_80k_cityscapes_bs_8.yaml)| 
|SETR_PUP   | ViT_Large |     8      |     40k   |   77.92   |       79.63        |  [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)     | [google](https://drive.google.com/file/d/12rMFMOaOYSsWd3f1hkrqRc1ThNT8K8NG/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1H8b3valvQ2oLU9ZohZl_6Q)(zmoi)    | [config](semantic_segmentation/configs/setr/SETR_PUP_Large_768x768_40k_cityscapes_bs_8.yaml)| 
|SETR_PUP   | ViT_Large |     8      |     80k   |   78.81   |       80.43     |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)    | [baidu](https://pan.baidu.com/s/1tkMhRzO0XHqKYM0lojE3_g)(f793)    | [config](semantic_segmentation/configs/setr/SETR_PUP_Large_768x768_80k_cityscapes_bs_8.yaml)| 
|SETR_MLA   | ViT_Large |     8      |     40k   |   76.70    |       78.96      |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)    | [baidu](https://pan.baidu.com/s/1sUug5cMKSo6mO7BEI4EV_w)(qaiw)    | [config](semantic_segmentation/configs/setr/SETR_MLA_Large_768x768_40k_cityscapes_bs_8.yaml)| 
|SETR_MLA   | ViT_Large |     8      |     80k   |  77.26     |       79.27      |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)    | [baidu](https://pan.baidu.com/s/1IqPZ6urdQb_0pbdJW2i3ow)(6bgj)    | [config](semantic_segmentation/configs/setr/SETR_MLA_Large_768x768_80k_cityscapes_bs_8.yaml)| 


#### ADE20K ####
|Model      | Backbone  | Batch_size | Iteration | mIoU (ss) | mIoU (ms+flip) | Backbone_checkpoint | Model_checkpoint     |     ConfigFile  |
|-----------|-----------|------------|-----------|-----------|----------------|-----------------------------------------------|-----------------------------------------------------------------------|------------|
|SETR_Naive | ViT_Large |     16      |     160k   | 47.57   |      48.12        |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)    | [baidu](https://pan.baidu.com/s/1_AY6BMluNn71UiMNZbnKqQ)(lugq)   | [config](semantic_segmentation/configs/setr/SETR_Naive_Large_512x512_160k_ade20k_bs_16.yaml)| 
|SETR_PUP   | ViT_Large |     16      |     160k   |  49.12   |      49.51        |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)    | [baidu](https://pan.baidu.com/s/1N83rG0EZSksMGZT3njaspg)(udgs)    | [config](semantic_segmentation/configs/setr/SETR_PUP_Large_512x512_160k_ade20k_bs_16.yaml)| 
|SETR_MLA   | ViT_Large |     8      |     160k   |  47.80   |       49.34        |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)    | [baidu](https://pan.baidu.com/s/1L83sdXWL4XT02dvH2WFzCA)(mrrv)    | [config](semantic_segmentation/configs/setr/SETR_MLA_Large_512x512_160k_ade20k_bs_8.yaml)| 
|DPT        | ViT_Large |     16     |     160k   |  47.21   |       -        |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)      |[baidu](https://pan.baidu.com/s/1PCSC1Kvcg291gqp6h5pDCg)(ts7h)   |  [config](semantic_segmentation/configs/dpt/DPT_Large_480x480_160k_ade20k_bs_16.yaml)
|Segmenter  | ViT_Tiny  |     16     |     160k   |  38.45   |       -        |   TODO      |[baidu](https://pan.baidu.com/s/1nZptBc-IY_3PFramXSlovQ)(1k97)   |  [config](semantic_segmentation/configs/segmenter/segmenter_Tiny_512x512_160k_ade20k_bs_16.yaml)
|Segmenter  | ViT_Small |     16     |     160k   |  46.07   |       -        |   TODO      |[baidu](https://pan.baidu.com/s/1gKE-GEu7gX6dJsgtlvrmWg)(i8nv)   |  [config](semantic_segmentation/configs/segmenter/segmenter_small_512x512_160k_ade20k_bs_16.yaml)
|Segmenter  | ViT_Base  |     16     |     160k   |  49.08   |       -        |   TODO      |[baidu](https://pan.baidu.com/s/1qb7HEtKW0kBSP6iv-r_Hjg)(hxrl)   |  [config](semantic_segmentation/configs/segmenter/segmenter_Base_512x512_160k_ade20k_bs_16.yaml) |
|Segmenter  | ViT_Large  |     16     |     160k   |  51.82   |       -        |   TODO      |[baidu](https://pan.baidu.com/s/121FOwpsYue7Z2Rg3ZlxnKg)(wdz6)   |  [config](semantic_segmentation/configs/segmenter/segmenter_Tiny_512x512_160k_ade20k_bs_16.yaml)
|Segmenter_Linear  | DeiT_Base |     16     |     160k   |  47.34   |       -        |   TODO      |[baidu](https://pan.baidu.com/s/1Hk_zcXUIt_h5sKiAjG2Pog)(5dpv)   |  [config](semantic_segmentation/configs/segmenter/segmenter_Base_distilled_512x512_160k_ade20k_bs_16.yaml)
|Segmenter  | DeiT_Base |     16     |     160k   |  49.27   |       -        |   TODO      |[baidu](https://pan.baidu.com/s/1-TBUuvcBKNgetSJr0CsAHA)(3kim)   |  [config](semantic_segmentation/configs/segmenter/segmenter_Base_distilled_512x512_160k_ade20k_bs_16.yaml) |
|Segformer  | MIT-B0 |     16     |     160k   |  38.37   |       -        |   TODO      |[baidu](https://pan.baidu.com/s/1WOD9jGjQRLnwKrRYzgBong)(ges9)   |  [config](semantic_segmentation/configs/segformer/segformer_mit-b0_512x512_160k_ade20k.yaml) |
|Segformer  | MIT-B1 |     16     |     160k   |  42.20   |       -        |   TODO      |[baidu](https://pan.baidu.com/s/1aiSBXMd8nP82XK7sSZ05gg)(t4n4)   |  [config](semantic_segmentation/configs/segmenter/segformer_mit-b1_512x512_160k_ade20k.yaml) |
|Segformer  | MIT-B2 |     16     |     160k   |  46.38   |       -        |   TODO      |[baidu](https://pan.baidu.com/s/1wFFh-K5t46YktkfoWUOTAg)(h5ar)   |  [config](semantic_segmentation/configs/segmenter/segformer_mit-b2_512x512_160k_ade20k.yaml) |
|Segformer  | MIT-B3 |     16     |     160k   |  48.35   |       -        |   TODO      |[baidu](https://pan.baidu.com/s/1IwBnDeLNyKgs-xjhlaB9ug)(g9n4)   |  [config](semantic_segmentation/configs/segmenter/segformer_mit-b3_512x512_160k_ade20k.yaml) |
|Segformer  | MIT-B4 |     16     |     160k   |  49.01   |       -        |   TODO      |[baidu](https://pan.baidu.com/s/1a25fCVlwJ-1TUh9HQfx7YA)(e4xw)   |  [config](semantic_segmentation/configs/segmenter/segformer_mit-b4_512x512_160k_ade20k.yaml) |
|Segformer  | MIT-B5 |     16     |     160k   |  49.73   |       -        |   TODO      |[baidu](https://pan.baidu.com/s/15kXXxKEjjtJv-BmrPnSTOw)(uczo)   |  [config](semantic_segmentation/configs/segmenter/segformer_mit-b5_512x512_160k_ade20k.yaml) |
| UperNet  | Swin_Tiny |     16     |     160k   |  44.90   |       45.37     |   -      |[baidu](https://pan.baidu.com/s/1S8JR4ILw0u4I-DzU4MaeVQ)(lkhg)   |  [config](semantic_segmentation/configs/upernet_swin/upernet_swin_tiny_patch4_windown7_512x512_160k_ade20k.yaml) |
| UperNet  | Swin_Small |     16     |     160k   |  47.88   |       48.90      |   -      |[baidu](https://pan.baidu.com/s/17RKeSpuWqONVptQZ3B4kEA)(vvy1)   |  [config](semantic_segmentation/configs/upernet_swin/upernet_swin_small_patch4_windown7_512x512_160k_ade20k.yaml) |
| UperNet  | Swin_Base |     16     |     160k   |   48.59   |       49.04      |   -      |[baidu](https://pan.baidu.com/s/1bM15KHNsb0oSPblQwhxbgw)(y040)   |  [config](semantic_segmentation/configs/upernet_swin/upernet_swin_base_patch4_windown7_512x512_160k_ade20k.yaml) |
| UperNet  | CSwin_Tiny |     16     |     160k   |  49.46   |           |[baidu](https://pan.baidu.com/s/1ol_gykZjgAFbJ3PkqQ2j0Q)(l1cp) | [baidu](https://pan.baidu.com/s/1gLePNLybtrax9yCQ2fcIPg)(y1eq)  |  [config](seman}tic_segmentation/configs/upernet_cswin/upernet_cswin_tiny_patch4_512x512_160k_ade20k.yaml) |
| UperNet  | CSwin_Small |     16     |     160k   |  50.88   |      | [baidu](https://pan.baidu.com/s/1mSd_JdNS4DtyVNYxqVobBw)(6vwk)   | [baidu](https://pan.baidu.com/s/1a_vhHoib0-BcRwTnnSVGWA)(fz2e)   | [config](semantic_segmentation/configs/upernet_cswin/upernet_cswin_small_patch4_512x512_160k_ade20k.yaml) |
| UperNet  | CSwin_Base |     16     |     160k   |  50.64   |      | [baidu](https://pan.baidu.com/s/1suO0jX_Tw56CVm3UhByOWg)(0ys7)   | [baidu](https://pan.baidu.com/s/1Ym-RUooqizgUDEm5jWyrhA)(83w3)   | [config](semantic_segmentation/configs/upernet_cswin/upernet_cswin_base_patch4_512x512_160k_ade20k.yaml) |

#### Trans10kV2 ####
|Model      | Backbone  | Batch_size | Iteration | mIoU (ss) | mIoU (ms+flip) | Backbone_checkpoint | Model_checkpoint     |     ConfigFile  |
|-----------|-----------|------------|-----------|-----------|----------------|-----------------------------------------------|-----------------------------------------------------------------------|------------|
|Trans2seg_Medium | Resnet50c |     16      |    16k    |  75.97  |      -        |   [google](https://drive.google.com/file/d/1C6nMg6DgQ73wzF21UwDVxmkcRTeKngnK/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1hs0tbSGIeMLLGMq05NN--w)(4dd5)    | [google](https://drive.google.com/file/d/1C6nMg6DgQ73wzF21UwDVxmkcRTeKngnK/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1wdOUD6S8QGqD6S-98Yb37w)(w25r)   | [config](semantic_segmentation/configs/trans2seg/Trans2Seg_medium_512x512_16k_trans10kv2_bs_16.yaml)| 

### GAN ###
| Model                          | FID | Image Size | Crop_pct | Interpolation | Model        |
|--------------------------------|-----|------------|----------|---------------|--------------|
| styleformer_cifar10            |2.73 | 32         | 1.0      | lanczos       |[google](https://drive.google.com/file/d/1iW76QmwbYz6GeAPQn8vKvsG0GvFdhV4T/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Ax7BNEr1T19vgVjXG3rW7g)(ztky)  |
| styleformer_stl10              |15.65| 48         | 1.0      | lanczos       |[google](https://drive.google.com/file/d/15p785y9eP1TeoqUcHPbwFPh98WNof7nw/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1rSORxMYAiGkLQZ4zTA2jcg)(i973)|
| styleformer_celeba             |3.32 | 64         | 1.0      | lanczos       |[google](https://drive.google.com/file/d/1_YauwZN1osvINCboVk2VJMscrf-8KlQc/view?usp=sharing)/[baidu](https://pan.baidu.com/s/16NetcPxLQF9C_Zlp1SpkLw)(fh5s) |
| styleformer_lsun               | 9.68 | 128        | 1.0      | lanczos       |[google](https://drive.google.com/file/d/1i5kNzWK04ippFSmrmcAPMItkO0OFukTd/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1jTS9ExAMz5H2lhue4NMV2A)(158t)|
> *使用**fid50k_full**指标在 Cifar10, STL10, Celeba 以及 LSUNchurch 数据集上评估结果.


## 图像分类的快速示例
如果需要使用模型预训练权重，需要转到对应子文件夹，例如， `/image_classification/ViT/`, 然后下载 `.pdparam` 权重文件并在python脚本中更改相关文件路径。模型的配置文件位于`.、configs/`.  

假设下载的预训练权重文件存储在`./vit_base_patch16_224.pdparams`, 在python中使用`vit_base_patch16_224`模型:
```python
from config import get_config
from visual_transformer import build_vit as build_model
# config files in ./configs/
config = get_config('./configs/vit_base_patch16_224.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./vit_base_patch16_224')
model.set_dict(model_state_dict)
```
> :robot: 详细用法庆参见每个模型对应文件夹中的README文件.


### 评估 ###
如果在单GPU上评估ViT模型在ImageNet2012数据集的性能，请使用命令行运行以下脚本：
```shell
sh run_eval.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg=./configs/vit_base_patch16_224.yaml \
    -dataset=imagenet2012 \
    -batch_size=16 \
    -data_path=/path/to/dataset/imagenet/val \
    -eval \
    -pretrained=/path/to/pretrained/model/vit_base_patch16_224  # .pdparams is NOT needed
```

<details>

<summary>
使用多GPU运行评估
</summary>


```shell
sh run_eval_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
    -cfg=./configs/vit_base_patch16_224.yaml \
    -dataset=imagenet2012 \
    -batch_size=16 \
    -data_path=/path/to/dataset/imagenet/val \
    -eval \
    -pretrained=/path/to/pretrained/model/vit_base_patch16_224  # .pdparams is NOT needed
```

</details>


### 训练 ###
如果使用单GPU在ImageNet2012数据集训练ViT模型，请使用命令行运行以下脚本：
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
  -cfg=./configs/vit_base_patch16_224.yaml \
  -dataset=imagenet2012 \
  -batch_size=32 \
  -data_path=/path/to/dataset/imagenet/train
```


<details>

<summary>
使用多GPU运行训练：
</summary>


```shell
sh run_train_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
    -cfg=./configs/vit_base_patch16_224.yaml \
    -dataset=imagenet2012 \
    -batch_size=16 \
    -data_path=/path/to/dataset/imagenet/train
```

</details>



## 贡献 ##
* 我们鼓励并感谢您对 **PaddleViT** 项目的贡献, 请查看[CONTRIBUTING.md](./CONTRIBUTING.md)以参考我们的工作流程和代码风格.  


## 许可 ##
* 此 repo 遵循 Apache-2.0 许可. 

## 联系 ##
如果您有任何问题, 请在我们的Github上创建一个[issue](https://github.com/BR-IDL/PaddleViT/issues).

