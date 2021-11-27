import os

g = os.walk("imagenet/train")
for path, dir_list, file_list in g:
    for file in file_list:
        new_dir = os.path.join(path, file.split('.')[0])
        if os.path.exists(new_dir):
            continue
        os.mkdir(new_dir)
        os.system(f'tar -xvf {os.path.join(path, file)} -C {new_dir}')