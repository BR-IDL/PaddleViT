English | [简体中文](./paddlevit-export-cn.md)

## PaddleViT: How to export model for deployment?
This document presents **how to use paddlevit code** to generate models for production deployment on server.

`PaddleViT` generates the inference model for server using Paddle Inference framework, which is designed and optimized for deploying models which can be excecuted on multiple platforms. Details about paddle inference can be found [here](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)

> Detailed official `paddle.inference` docs and examples can be found: [here](https://paddle-inference.readthedocs.io/en/latest/index.html)

### 1. How to export model in PaddleViT?
In `PaddleViT`, export model is easy and straightforward. The general idea is to export and save the model from dygraph to static mode. Optimization settings can be customized according to users needs, such as fuse operations or enable inplacement. We have provided a [script](../image_classification/BEiT/export_models.py) to help start your implementations for exporting models :

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

> **Note**: you might see errors running the above script using other models in paddlevit. On most case, the error is caused by the tensor indexing (from ops such as reshape, flatten, transpose, etc), or the extra operations applied in model's forward method (such as using ops from paddle.nn.functional in inference).  The solution can be various depending on the specific model implementations, but the general idea to solve the problems are:
> 1. Replace the tensor shape op (e.g., a.shape[0]) to `paddle.shape` (e.g., paddle.shape(x)[0]) , avoid use more than one `-1` in tensor broadcasting ops (such as reshape)
> 2. Replace the `paddle.nn.functional` operations used in `forward` method by defining those ops using `paddle.nn` layers in `__init__`.

> If you have met any problems exporting the paddlevit models, please raise an issue in our github repo.

### 2. Run the inference of exported model using paddle inference api (python)
To test if the exported model, we create an python script which uses paddle inference api to run the inference for expored model, such as:
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

If you want to check the correctness of the model output given real images as inputs, uncomment the "read images from file" part in the above script, which will enable the model do the inference using images as the inputs. Please note that, in the `load_img` method, you will have to use the same data preprocessing operations as the original model.

> C++ examples can be found in officail paddle inference docs: [here](https://paddle-inference.readthedocs.io/en/latest/quick_start/cpp_demo.html)