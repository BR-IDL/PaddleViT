[English](./paddlevit-quant-en.md) | 简体中文

## PaddleViT: 如何使用PaddleSlim进行量化?
本文档描述了如何使用**PaddleViT**和PaddleSlim工具进行模型量化。

使用`PaddleViT` 的模型用于PaddleLite等工具进行端侧部署时，我们使用PaddleSlim生成量化模型，该框架提供方便、优化的模型工具和相关组件，并支持多个平台，更多介绍请见[这里](https://paddleslim.readthedocs.io/zh_CN/develop/intro.html)



### PaddleViT如何生成静态模型并使用量化处理?
在`PaddleViT`中，我们提供了非常简单的预测模型生成方法。其核心思想是使用paddle api将动态图模型转换为静态图模式，并储存相关模型到文件系统。在这一过程中我们可以设置许多优化配置，例如合并操作，允许inplace操作等。在此，我们提供了一个简单的[脚本文件](../image_classification/T2T_ViT/export_models.py)，您可以参考并实现自己的模型转换代码：


```python
import os
import paddle
from config import get_config
from t2t_vit import build_t2t_vit as build_model


def main():
    """load .pdparams model file, save to static model:
        .pdiparams, .pdiparams.info, .pdmodel
    """
    model_name = "t2t_vit_7"
    model_folder = './'
    out_folder = './t2t_vit_7_static'
    os.makedirs(out_folder, exist_ok=True)

    model_path = f'{model_folder}/{model_name}.pdparams'
    out_path = f'{out_folder}/inference'
    # STEP1: load config file and craete model
    config = get_config(f'./configs/{model_name}.yaml')
    model = build_model(config)
    # STEP2: load model weights 
    model_state = paddle.load(model_path)
    if 'model' in model_state:
        if 'model_ema' in model_state:
            model_state = model_state['model_ema']
        else:
            model_state = model_state['model']
    model.set_state_dict(model_state)    
    model.eval()

    # STEP3: craete export build strategy
    build_strategy = paddle.static.BuildStrategy()
    # some optimized settings
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.reduce_strategy = paddle.static.BuildStrategy.ReduceStrategy.Reduce
    build_strategy.fuse_broadcast_ops = True
    build_strategy.fuse_elewise_add_act_ops = True

    # STEP4: export model to static 
    img_size = [config.DATA.IMAGE_CHANNELS, config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE]
    model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec(shape=[None] + img_size, dtype='float32'), name='x'],
        build_strategy=build_strategy)

    # STEP5: save static model to file
    paddle.jit.save(model, out_path)
    print(f'export model {model_name} done')


if __name__ == "__main__":
    main()

```

> **注意**: 在模型转换过程中，您可能会遇到程序报错。在多数情况下，错误是由于（1）在reshape等操作中使用了动态的取值，例如在forward的操作需要使用input tensor的尺寸等信息，或者在broadcasting操作中使用多于1个的`-1`，例如reshape中使用多个-1。（2）在forward中使用了paddle.nn.functional操作。由于问题取决于您使用的模型，我们无法提供完整的解决方案，通常的解决方法有：
> 1. 将tensor的取shape操作(e.g., a.shape[0]) 改为调用 `paddle.shape` (e.g., paddle.shape(x)[0]) , 在reshape等操作中避免使用多于1个的`-1`。
> 2. 将`forward`方法中的`paddle.nn.functional` 操作，改为在模型`__init__`方法中定义`paddle.nn`层后再在`forward`中调用，例如将`paddle.nn.functional.softmax`改为(1)在`__init__`中定义`self.softmax = paddle.nn.Softmax()`; (2)在`forward`中调用`out = self.softmax(in)`

静态模型将会输出保存到对应路径：
```
t2t_vit_7_static/
├── inference.pdiparams
├── inference.pdiparams.info
└── inference.pdmodel
```
> 如果您在模型输出时遇到任何问题， 请在我们PaddleViT的github repo中提issue以便我们尽快解决问题。


### 2. 使用PaddleSlim 进行模型量化（离线）
我们可以使用paddleslim相关脚本对模型进行量化并测试性能，该部分详细使用方法请参考[PaddleSlim文档](https://paddleslim.readthedocs.io/zh_CN/develop/quick_start/dygraph/dygraph_quant_post_tutorial.html)




#### PaddleSlim提供了`quant_post.py`脚本方便我们将静态模型进行量化（离线）处理
> 参考链接：[here](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/demo/quant/quant_post/quant_post.py)

我们使用命令行调用该脚本并设置相关参数完成量化：
```shell
python quant_post.py --model_path ./t2t_vit_7_static --save_path ./t2t_vit_7_quant --input_name x --batch_size 32
```
如果运行无误，量化模型将保存到对应路径：
```
t2t_vit_7_quant/
├── __model__
└── __params__
```

#### PaddleSlim同时提供了`eval.py`脚本方便我们对量化（离线）模型进行评估
> 参考链接：[here](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/demo/quant/quant_post/eval.py)

我们使用命令行调用该脚本并设置相关参数完成评估：
```shell
python eval.py --model_path quant_t2t/ --batch_size 32
```

> 更多PaddleSlim相关文档可参考：[here](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_post)