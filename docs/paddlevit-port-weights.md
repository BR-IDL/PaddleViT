English | [简体中文](./paddlevit-port-weights-cn.md)

## PaddleViT: How to port model from Pytorch to Paddle?
> Sample code: [here](../image_classification/SwinTransformer/port_weights/load_pytorch_weights.py)

### Step 0:
We assume you are trying to implement your Paddle version of some ViT model, from some PyTorch implementations. You want to port the pretrained weights from pytorch `.pth` file to paddle `.pdparams` file.

So we now have:
- One `torch.nn.Module` class implements the model in pytorch.
- One `.pth` pretrained weight file corresponding to the pytorch model.
- One `paddle.nn.Layer` class we implemented the same model in paddle.

> Note: the `paddle.nn.Layer` class must implemented in the similar way of your refferred `torch.nn.Module`. Here 'similar' means the param sizes, tensor shapes, and compute logics are the same, while the name of the layers/params or the detailed implementations could be different.

We still need to implement:
- `load_pytorch_weights.py`, contains the methods and name mappings for model conversion.

Now we show how to implement `load_pytorch_weights.py`.

### Step 1:
 Load your paddle model, e.g.:
 ```python
 paddle_model = build_model(config)
 paddle_model.eval()
 ```
 You can just init a model class to build a model object, please refer to our PPViT code for detailed model definitions and usage of `config`.


### Step 2:
 Load your pytorch model with pretrained weights. 
 
 For example, if we use models from `timm` project:
 ```python
 import timm
 torch_model = timm.create_model('vit_base_patch16_224', pretrained=True)
 torch_model.eval()
 ```
> timm: https://github.com/rwightman/pytorch-image-models

### Step 3:
 Check the name mappings (**manually**).
 In `torch_to_paddle_mapping` method, you create a list of string tuples defines the corresponding param and buffer names for torch and paddle models. E.g.:
- In **torch** model one param with name `patch_embed.proj.weight` 
- In **paddle** model, same param is named `embeddings.patch_embeddings.weight`
Then you have a tuple `(patch_embed.proj.weight, embeddings.patch_embeddings.weight)` saved in the mapping list.

 > NOTE: You can use **for loop** and **prefix strings** to semi-automate your name mapping process.

 > NOTE: Do NOT forget to add name mappings for `model.named_buffers()`

Usually I copy the printed torch param/buffer names and shapes, and printed paddle param/buffer names and shapes, each in an individual text file, then check the mapping line by line and modify the `torch_to_paddle_mapping` if necessary.

If all the name mappings are correct, run the conversion by:
```python
paddle_model = convert(torch_model, paddle_model)
```
> This method will convert the param weights from torch to the proper format, and then set the value to corresponding paddle params. The returned object is the paddle model obbject with pretrained weights same as pytorch model.

> In `convert` method, weights of `torch.nn.Linear` is applied a `transpose`, to match the weights shape of `paddle.nn.Linear`.
### Step 4:
Check correctness. 
Create a batch data corresponding to the mode input, e.g. :
```python
# check correctness
x = np.random.randn(2, 3, 224, 224).astype('float32')
x_paddle = paddle.to_tensor(x)
x_torch = torch.Tensor(x).to(device)
```
Then do inference and convert output into numpy array:
```
out_torch = torch_model(x_torch)
out_paddle = paddle_model(x_paddle)

out_torch = out_torch.data.cpu().numpy()
out_paddle = out_paddle.cpu().numpy()
```
Finally, check if the outputs are same for `paddle_model` and `torch_model`:
```python
assert np.allclose(out_torch, out_paddle, atol = 1e-5)
```

### Step 5:
Save model weights for paddle:
```python
paddle.save(paddle_model.state_dict(), model_path)
```

> **Tips:**
> - BN layers usually have buffers such as `_mean`, and `_variance`
> - Do not forget customized buffer defined in model, e.g., `paddle.register_buffer()`
> - Use batched data  (batchsize > 1) to test results.
> - Some params are 2-D but non Linear params, so `_set_value` must set `transpose=False`.
