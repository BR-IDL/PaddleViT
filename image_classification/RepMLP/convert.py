import argparse
import paddle
import os
from repmlp import repmlp_model_convert
from config import get_config
from repmlp_resnet import build_repmlp_resnet as build_model

parser = argparse.ArgumentParser(description='RepMLP_ResNet Conversion')
parser.add_argument('--load_path', help='path to the weights file')
parser.add_argument('--save_path', help='path to the weights file')
parser.add_argument('--arch', default='RepMLP-Res50-light-224', help='convert architecture')

def convert():
    args = parser.parse_args()
    if args.arch == 'RepMLP-Res50-light-224':
        config = get_config('./configs/repmlpres50_light_224_train.yaml')
        train_model = build_model(config)
    else:
        raise ValueError('TODO')

    if os.path.isfile(args.load_path):
        print("=> loading checkpoint '{}'".format(args.load_path))
        train_model.set_state_dict(paddle.load(args.load_path))
        print("=> loading done")
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    print("=> convert training to deploy ...")
    repmlp_model_convert(train_model, save_path=args.save_path)


if __name__ == '__main__':
    convert()