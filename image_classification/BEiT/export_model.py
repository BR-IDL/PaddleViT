import paddle
from config import get_config
from beit import build_beit as build_model


def main():
    model_name = 'beit_base_patch16_224'
    model_path = './beit_base_patch16_224.pdparams'
    out_path = './beit_base'
    config = get_config(f'./configs/{model_name}.yaml')
    model = build_model(config)
    model_state = paddle.load(model_path)
    if 'model' in model_state:
        if 'model_ema' in model_state:
            model_state = model_state['model_ema']
        else:
            model_state = model_state['model']
    model.set_state_dict(model_state)

    model.eval()

    img_size = [config.DATA.IMAGE_CHANNELS, config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE]
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.reduce_strategy = paddle.static.BuildStrategy.ReduceStrategy.Reduce
    build_strategy.fuse_broadcast_ops = True
    build_strategy.fuse_elewise_add_act_ops = True

    model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec(shape=[None] + img_size, dtype='float32')],
        build_strategy=build_strategy)

    paddle.jit.save(model, out_path)
    print('export model done')


if __name__ == "__main__":
    main()

