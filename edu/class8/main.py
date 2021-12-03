import argparse
from config import get_config
from config import update_config

def get_arguments():
    parser = argparse.ArgumentParser('ViT')
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-batch_size', type=int, default=None)

    arguments = parser.parse_args()
    return arguments

def main():
    cfg = get_config()
    print(cfg)
    print('-----')
    cfg = get_config('./a.yaml')
    print(cfg)
    print('-----')
    args = get_arguments()
    cfg = update_config(cfg, args)
    print(cfg)

if __name__ == "__main__":
    main()
