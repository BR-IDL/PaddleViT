import paddle
from coco import build_coco
from coco import get_loader
from detr import build_detr

def main():
    model, criterion, postprocessors = build_detr()
    model_state = paddle.load('./detr_resnet50.pdparams')
    model.set_dict(model_state)
    model.eval()

    # 2. Create val dataloader
    dataset_val = build_coco('val', '/dataset/coco/')
    dataloader_val = get_loader(dataset_val,
                                batch_size=4,
                                mode='val',
                                multi_gpu=False)

    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader_val):
            samples = data[0]
            targets = data[1]
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            orig_target_sizes = paddle.stack([t['orig_size'] for t in targets], axis=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            res = {target['image_id']: output for target, output in zip(targets, results)}

            print(f'{batch_id}/{len(dataloader_val)} done')

    print('all done')



if __name__ == '__main__':
    main()
