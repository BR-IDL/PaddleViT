# 如何使用PaddleViT的分类模型进行Predict？
> 由于无法确定预测数据集的格式以及具体的预测目标，当前版本的PaddleViT分类模型只提供了训练和验证代码，对应数据集包含图像数据和标签，具体数据的存放形式可以参考 [这里](https://github.com/BR-IDL/PaddleViT/tree/develop/image_classification#data-preparation)。 本教程提供简单的流程，帮助用户使用PaddleViT进行模型预测。
## 当使用PaddleViT对某数据进行预测时，我们需要修改以下几个文件：
- `dataset.py`: 添加目标数据集的加载方式
- `main_single_gpu.py` / `main_multi_gpu.py`: 添加predict方法
- `run_test.sh`: 新建shell脚本启动预测程序


## 1. 在`dataset.py`中添加数据加载方式
### 1.1 `dataset.py` 数据加载基本逻辑:
PaddleViT分类模型采用统一的数据加载方式：
```python
from datasets import get_dataset
your_dataset = get_dataset(config, mode='train')
your_dataloader = get_dataloader(config, dataset, mode='train', multi_process=False)
```
在得到`your_dataloader`之后，可以使用常见的for循环方式加载数据开始训练或测试：
```python
for batch_id, batch_data in enumerate(your_dataloder):
    # start your training/validation 
    ...
```
> config的基本概念参考[这里](https://github.com/BR-IDL/PaddleViT/blob/develop/docs/paddlevit-config.md)

### 1.2 自定义数据集
当需要使用自定义数据集时。我们需要：
1. 在`datasets.py`（或者新建python文件）中创建`YOUR_DATASET`类，继承`paddle.io.Dataset`父类，并且实现`__len__`和`__getitem__`方法：
    ```python
    class YOUR_DATASET(paddle.io.dataset):
        def __init__(self, file_folder, mode, transform=None):
            super().__init__()
            # TODO: load image list (if train, load corresponding label info)
            ...
        def __len__(self):
            # TODO: return num of data
        def __getitem__(self, index):
            # TODO: return image data given index, if transform is not none, apply transform first.
    ```
    > 注意： 在`__getitem__`方法返回前可能需要先对数据进行预处理。
2. 在`datasets.py`中修改数据处理方法`get_train_transforms` 和 `get_val_transforms`，也可以新建自己的transforms，例如
`get_test_transforms`：
    ```python
    def get_test_transform(config):
        scale_size = int(math.floor(config.DATA.IMAGE_SIZE / config.DATA.CROP_PCT))
        transforms_test = transforms.Compose([
            transforms.Resize(scale_size, 'bicubic'), # single int for resize shorter side of image
            transforms.CenterCrop((config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.DATA.IMAGENET_MEAN, std=config.DATA.IMAGENET_STD),
        ])
        return transforms_test
    ```
3. 在`datasets.py`中的`get_dataset`方法增加该数据集的调用方法:
    ```python
    def get_dataset(config, mode='train'):
        assert mode in ['train', 'val', 'test']
        ...
        # ADD your dataset for testing:
        elif config.DATA.DATASET == "your_dataset_name":
            dataset = YOUR_DATASET(config.DATA.DATA_PATH,
                                   mode='test',
                                   transform=get_test_transforms(config))
        ...
        return dataset
    ```
    

## 2. 在`main_single_gpu.py` / `main_multi_gpu.py`中添加predict方法：

### 2.1 添加`predict`方法
我们可以在主程序中，增加预测方法（`predict`），和`validate`方法的区别是，`predict`方法不会加载到标签信息，所以不会返回acc等指标，仅返回预测结果,以`main_single_gpu.py`为例，我们可以增加以下方法：
```python
def predict(dataloader, model, total_batch, debug_steps=100, logger=None):
    """Predict for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        logger: logger for logging, default: None
    Returns:
        preds: prediction results
        pred_time: float, prediction time
    """
    model.eval()
    time_st = time.time()

    preds = []
    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            image = data[0]
            #label = data[1] # no label info in prediction
            output = model(image)
            #loss = criterion(output, label) # no criterion in prediction
            pred = F.softmax(output)
            # no acc in prediction
            #acc1 = paddle.metric.accuracy(pred, label.unsqueeze(1))
            #acc5 = paddle.metric.accuracy(pred, label.unsqueeze(1), k=5)
            preds.append(pred)
            if logger and batch_id % debug_steps == 0:
                logger.info(f"Pred Step[{batch_id:04d}/{total_batch:04d}], done")

    pred_time = time.time() - time_st
    return preds, pred_time
```
返回结果为`list`, 包含每个batch的预测结果（`tensor`）和预测时间

### 2.2 修改主程序
在`main`方法中，我们可以修改`eval`相关部分用以完成prediction:
```python
def main():
    ...
    # STEP 2: Create train and val dataloader
    ...
    dataset_test = get_dataset(config, mode='test')
    dataloader_test = get_dataloader(config, dataset_test, 'test', False)
    ...
    # STEP 7: Prediction 
    if config.EVAL:
        logger.info('----- Start Prediction')
        preds, pred_time = predict(
            dataloader=dataloader_val,
            model=model,
            total_batch=len(dataloader_val),
            debug_steps=config.REPORT_FREQ,
            logger=logger)
        
        # TODO: do your processing on predictions
        return 
```

## 3. 新建shell脚本启动预测程序`run_test.sh`:
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='your_dataset_name' \
-batch_size=128 \
-data_path='your_dataset_path' \
-eval \
-pretrained='./vit_base_patch16_224'
```