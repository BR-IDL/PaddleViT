简体中文 | [English](./paddlevit-port-weights.md)

## PaddleViT: 如何将模型从 Pytorch 移植到 Paddle?
> 源码: [here](../image_classification/ViT/load_pytorch_weights.py)

### Step 0:
如果你想要从一些ViT模型的PyTorch实现转换到Paddle版本，并需要将预训练权重从pytorch `.pth`文件转换为paddle`.pdparams` 文件。

首先需要具有的要素:
- 一个 `torch.nn.Module` 类在pytorch中实现模型.
- 一个与Pytorch模型对应的 `.pth` 预训练权重文件.
- 一个在paddle中实现相同模型的 `paddle.nn.Layer` 类.

> 注意:  `paddle.nn.Layer`类必须以与你引用的 `torch.nn.Module`相似的方式实现. 此处的'similar' 表示参数大小、张量形状和计算逻辑相同，而层/参数的名称或详细实现可能不同。

我们还需要实现:
- `load_pytorch_weights.py`, 包含模型转换和名称映射的方法.

接下来我们展示如何实现 `load_pytorch_weights.py`.

### Step 1:
加载paddle模型, 例如:
 ```python
 paddle_model = build_model(config)
 paddle_model.eval()
 ```
 你可以只初始化一个模型类用于构建一个模型对象，详细的模型定义和`config`用法请参考我们的PPViT代码。


### Step 2:
加载你的pytorch模型的预训练权重。

 例如，如果我们使用来自 `timm` 项目的模型:
 ```python
 import timm
 torch_model = timm.create_model('vit_base_patch16_224', pretrained=True)
 torch_model.eval()
 ```
> timm: https://github.com/rwightman/pytorch-image-models

### Step 3:
检查名称映射(**手动**).
在 `torch_to_paddle_mapping` 方法中，你创建了一个字符串元组列表，定义了torch和paddle模型的相应参数和缓冲区名称，例如：
- 在**torch** 模型中，一个名为`patch_embed.proj.weight` 的参数
- 在 **paddle** 模型中, 相同的参数被命名为 `embeddings.patch_embeddings.weight`
然后你有一个元组 `(patch_embed.proj.weight, embeddings.patch_embeddings.weight)` 保存在映射列表中。

 > 注意: 你可以使用 **for loop** 和 **prefix strings** 来半自动化你的名称映射过程。
 > 注意: 不要忘记为`model.named_buffers()`添加名称映射

通常我们会打印torch 参数/缓存区的名称和形状，并打印paddle 参数/缓冲区的名称和形状，每个都在单独的文本文件中，然后逐行检查映射，并在必要时修改 `torch_to_paddle_mapping`.

如果所有名称映射都正确，请通过以下方式运行转换：
```python
paddle_model = convert(torch_model, paddle_model)
```
> 此方法见torch中的参数权重转化为正确格式，然后将值设置为对应的paddle参数。返回的对象是具有与pytorch模型相同的预训练权重的paddle模型对象。

> 在 `convert`方法中， `torch.nn.Linear`的权重应用于 `transpose`, 用于匹配 `paddle.nn.Linear`权重的维度.
### Step 4:
检查正确性。

创建与模型输入相对应的批处理数据，例如：

```python
# check correctness
x = np.random.randn(2, 3, 224, 224).astype('float32')
x_paddle = paddle.to_tensor(x)
x_torch = torch.Tensor(x).to(device)
```
然后进行推理，将输出转换为numpy数组：
```
out_torch = torch_model(x_torch)
out_paddle = paddle_model(x_paddle)

out_torch = out_torch.data.cpu().numpy()
out_paddle = out_paddle.cpu().numpy()
```
最后, 检查`paddle_model` 和 `torch_model`的输出是否相同:
```python
assert np.allclose(out_torch, out_paddle, atol = 1e-5)
```

### Step 5:
保存paddle的模型权重：
```python
paddle.save(paddle_model.state_dict(), model_path)
```

> **提示:**
> - BN 层通常具有缓冲区，例如 `_mean`和 `_variance`
> - 不要忘记模型中定义的自定义缓冲区, 例如, `paddle.register_buffer()`
> - 使用批处理数据(batchsize > 1)来测试结果。
> - 一些参数是二维但非线形参数，所以`_set_value` 必须设置为 `transpose=False`.
