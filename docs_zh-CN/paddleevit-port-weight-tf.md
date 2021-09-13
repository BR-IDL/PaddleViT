## 如何将tensorflow 的checkpoint模型转换为paddle模型

### Step 0

首先我们假设你正在尝试参照tensorflow版本的代码实现一个paddle版本的ViT模型。你尝试将tensorflow的checkpoint模型转换为.pdparams模型。

所以我们必须拥有以下几种文件：

+ 一份基于tensorflow实现的模型代码。
+ 一份对应模型的checkpoint模型文件，包含model文件、model.data文件、model.index文件、model.meta文件。
+ 一份基于paddle.nn.Layer类实现对应模型。

> tensorflow框架可以直接将checkpoint模型当作一个static_dict读取出来，因此在转换模型时我们并不需要对应的tensorflow版本实现的模型源码。

接下来我们的任务便是实现load_tf_weight.py，并通过这份代码来完成模型文件之间的转换。在本文接下来的部分我们将一步一步的展示如何根据自己的需求完成对应的load_tf_weight.py文件。

### Step 1

首先你需要创建自己的paddle模型：

```
paddle_model = build_model(config)
paddle_model.eval()
```

具体的模型代码需要你根据自己想要复现的模型编写，如果需要的话可以直接适用PaddleVit中的相关源码。
> 例如当你需要实现window attention模块以及对应的relative position embedding机制，你完全可以参考PaddleVit中SwinTransformer/swin_transformer.py中的WindowAttention类。

### Step 2

使用Tensorflow读取对应的checkpoint模型，具体读取方式可以参考如下代码:
```
import numpy as np
from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader(your_path)
var_to_shape_map = reader.get_variable_to_shape_map()

static_dict = {}

for key in var_to_shape_map:
    key_list = key.split('/')
    if key_list[-1] == 'Adam' or key_list[-1] == 'Adam_1':
        continue
    static_dict[key] = np.array(reader.get_tensor(key))

np.save('yourmodel.npy', static_dict)

```
> 通过这份代码，你可以将tensorflow的checkpoint模型转存为numpy的.npy文件。
> 
> 其中需要注意的是tensorflow存取的checkpoint模型中，optimizer部分的参数和model的参数是混在一起的，因此在存取.npy文件时需要将optimizer部分的参数过滤掉。

### Step 3:

接下来你需要手动的检查你自己实现的paddle模型和对应的tensorflow模型之间的参数名字映射关系。为了更加详细观察两者之间的参数名称映射关系，你可能需要手动将模型中的所有参数打印一遍。打印paddle模型的参数你可以使用如下代码：

```
def print_model_named_params(model):
    for name, param in model.named_parameters():
        print(name, param.shape)

def print_model_named_buffers(model):
    for name, buff in model.named_buffers():
        print(name, buff.shape)

```

其中的*print_model_named_params*函数负责打印paddle模型中所有的named_parameters，而*print_model_named_buffers*函数负责打印paddle模型中所有的named_buffers。

在找到paddle模型和tensorflow模型之间的参数映射关系之后，你可以编写一个程序用来生成对应的映射列表。

在观察映射关系时需要注意以下几点：
> 1. 在tensorflow中conv层的卷积核参数名称为kernel,而在paddle中conv层的卷积核参数名称为weight。
> 2. 在tensorflow中conv2d层的卷积核参数值的维度信息为：[height, width, in_channels, out_channels],而在paddle中conv2d层的卷积核参数值的维度信息为:[out_channels, in_channels, height, width]。在转换之前一定要注意将对应参数使用*transpose()*进行转置。

在生成完对应的列表之后你可以使用如下的代码来完成模型的转换工作：
```
def convert(tf_static, paddle_model):
    def _set_value(tf_name, pd_name):
        tf_shape = tf_static[tf_name].shape
        pd_shape = tuple(pd_params[pd_name].shape)
        print(f'set {tf_name} {tf_shape} to {pd_name} {pd_shape}')
        value    = tf_static[tf_name]
        if len(value.shape) == 4:
            value = value.transpose((3, 2, 0, 1))
        if len(value.shape) == 5:
            value = value.transpose((4, 3, 0, 1, 2))
        pd_params[pd_name].set_value(value)

    global mapping
    pd_params = {}
    for name, param in paddle_model.named_parameters():
        pd_params[name] = param
    
    for name, param in paddle_model.named_buffers():
        pd_params[name] = param

    for tf_name, pd_name in mapping:
        _set_value(tf_name, pd_name)
    return paddle_model
```
> 在该函数中，参数tf_static应当为你之前转换的.npy文件的读取结果。该文件的读取你可以使用*np.load(youmodel.npy, allow_pickle=True)*函数，读取的结果为一个字典数据结构，里面存储着模型的参数名称以及其对应的tensor值。其中的tensor值通常情况下采用的是np.float32格式存储的np.array。


### Step 4:


接下来你需要验证转换模型的正确性。你可以通过生成一个随机的输入tensor，分别将其喂给paddle模型和tensorflow模型，你可能需要编写类似于下方示例的代码：
```
x  = np.random.randn(2, 3, 224, 224)
x_paddle = paddle.to_tensor(x)
```

然后分别执行两个模型的推理:
```
out_paddle = paddle_model(x_paddle).cup().numpy()
out_tf     = tf_model.valid_step(session, x, train=False)
```
> 注意不同版本的tensorflow模型实现方式不太一样，上面的例子展示的tensorflow模型推理代码仅仅适用于tensorflow1.4版本。

最后，你需要检查以下tensorflow模型的输出和paddle模型的输出是否一致:
```
assert np.allclose(out_paddle, out_tf, atol = 1e-5)
```

Step 5:
保存对应的paddle模型权重:
```
paddle.save(paddle_model.static_dict(), model_path)
```

