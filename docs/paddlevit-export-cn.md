[English](./paddlevit-export-en.md) | 简体中文

## PaddleViT: 如何部署?
本文档描述了如何使用**PaddleViT**生成可供部署的模型文件（server端）。

使用`PaddleViT` 的模型用于部署，我们使用Paddle Inference框架生成inference模型，该框架提供方便、优化的模型装换工具和部署相关组件，并支持多个平台，更多介绍请见[这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)


> `paddle.inference` 的官方文档和示例请参考[这里](https://paddle-inference.readthedocs.io/en/latest/index.html)

### 1. PaddleViT如何生成预测模型?
在`PaddleViT`中，我们提供了非常简餐的预测模型生成方法。其核心思想是使用paddle api将动态图模型转换为静态图模式，并储存相关模型到文件系统。在这一过程中我们可以设置许多优化配置，例如合并操作，允许inplace操作等。在此，我们提供了一个简单的[脚本文件](../image_classification/BEiT/export_models.py)，您可以参考并实现自己的模型转换代码：


```python
def main():
    """load .pdparams model file, save to static model:
        .pdiparams, .pdiparams.info, .pdmodel
    """
    model_names = [
        "beit_base_patch16_224",
        "beit_large_patch16_224",
    ]
    model_folder = './'
    out_folder = './'

    for model_name in model_names:
        model_path = f'{model_folder}/{model_name}.pdparams'
        out_path = f'{out_folder}/{model_name}'
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
            input_spec=[paddle.static.InputSpec(shape=[None] + img_size, dtype='float32')],
            build_strategy=build_strategy)

        # STEP5: save static model to file
        paddle.jit.save(model, out_path)
        print(f'export model {model_name} done')
```

> **注意**: 在模型转换过程中，您可能会遇到程序报错。在多数情况下，错误是由于（1）在reshape等操作中使用了动态的取值，例如在forward的操作需要使用input tensor的尺寸等信息，或者在broadcasting操作中使用多于1个的`-1`，例如reshape中使用多个-1。（2）在forward中使用了paddle.nn.functional操作。由于问题取决于您使用的模型，我们无法提供完整的解决方案，通常的解决方法有：
> 1. 将tensor的取shape操作(e.g., a.shape[0]) 改为调用 `paddle.shape` (e.g., paddle.shape(x)[0]) , 在reshape等操作中避免使用多于1个的`-1`。
> 2. 将`forward`方法中的`paddle.nn.functional` 操作，改为在模型`__init__`方法中定义`paddle.nn`层后再在`forward`中调用，例如将`paddle.nn.functional.softmax`改为(1)在`__init__`中定义`self.softmax = paddle.nn.Softmax()`; (2)在`forward`中调用`out = self.softmax(in)`

> 如果您在模型输出时遇到任何问题， 请在我们PaddleViT的github repo中提issue以便我们尽快解决问题。

### 2. 使用Paddle inference api验证预测模型 (python)
我们可以使用paddle inferece的python api方便快捷的验证转换的预测模型是否成功运行：
```python
import numpy as np
import paddle.inference as paddle_infer
from paddle.vision import image_load
from paddle.vision import transforms


def load_img(img_path, sz=224):
    """load image and apply transforms"""
    if isinstance(sz, int):
        sz = (sz, sz)
    data = image_load(img_path).convert('RGB')
    trans = transforms.Compose([
        transforms.Resize(sz, 'bicubic'),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
        ])
    data = trans(data)
    return data



def main():
    config = paddle_infer.Config(
        './beit_base_patch16_224.pdmodel',
        './beit_base_patch16_224.pdiparams',
    )
    batch_size = 4
    in_channels = 3
    image_size = 224

    predictor = paddle_infer.create_predictor(config)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    batch_data = np.random.randn(batch_size, in_channels, image_size, image_size).astype('float32')

    ## read images from file
    #data_all = []
    #for i in range(1, 5):
    #    image_path = f'./images/{i}.jpg'
    #    data = load_img(image_path, sz=image_size)
    #    # convert back to numpy, copy_from_cpu need numpy
    #    data = data.numpy()
    #    data_all.append(data)
    #batch_data = np.array(data_all)
    #print(batch_data.shape)
    

    input_handle.reshape([batch_size, in_channels, image_size, image_size])
    input_handle.copy_from_cpu(batch_data)
    
    predictor.run()
    
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()
    print(output_data)
    print('output shape = ', output_data.shape)


if __name__ == "__main__":
    main()
```
如果您需要使用真实图片来测试模型的准备度等，请将上面代码中的”read images from file“部分取消注释，该部分将从文件中读取图片并作为模型的输入进行推理预测。注意，您可能需要修改`load_img`中的数据处理部分，该部分必须与python模型定义的数据预处理（预测部分）保持一致。


> 如果您需要使用C++部署模型，请参考官方文档： [这里](https://paddle-inference.readthedocs.io/en/latest/quick_start/cpp_demo.html)