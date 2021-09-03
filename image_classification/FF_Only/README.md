# Pay Attention to MLPs, [arxiv](https://arxiv.org/abs/2105.02723) , [code](https://github.com/lukemelas/do-you-even-need-attention)

## Usage

```python
from config import get_config
from ffonly import build_ffonly
# config files in ./configs/
config = get_config('./configs/FF_Base.yaml')
# build model
model = build_ffonly(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('/home/aistudio/data/data96150/linear_base.pdparams')
model.set_dict(model_state_dict)
```

## Models Zoo

|  Model  | Top-1 Acc. | Top-5 Acc. |                             Link                             |
| :-----: | :--------: | :--------: | :----------------------------------------------------------: |
| FF_Base |   0.746    |   0.916    | [AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/96150) |
| FF_Tiny |   0.609    |   0.837    | [AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/96150) |

