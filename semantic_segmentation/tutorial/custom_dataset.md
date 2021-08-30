# Custom DataSet

In order to train on your own dataset, you should do the following three steps:

## Data Preparation

### Data Structure

To be simple, you should reorganize your dataset as follow. 

```none
├── your_dataset
│   ├── images
│   │   ├── training
│   │   │   ├── xxx{img_suffix}
│   │   │   ├── yyy{img_suffix}
│   │   │   ├── zzz{img_suffix}
│   │   ├── validation
│   ├── annotations
│   │   ├── training
│   │   │   ├── xxx{seg_map_suffix}
│   │   │   ├── yyy{seg_map_suffix}
│   │   │   ├── zzz{seg_map_suffix}
│   │   ├── validation

```

Images and labels are stored separately, and are part into training and testing set. The above four directory path will be specific in the script, in a relative way to the dataset root path, so you can rename them as you like.

### Annotations Format

Only support for gray-scale image now, you should always transform your image_label into gray-scale image first if needed.

The pixel intensity means the class index, and you can set a ignore_index which doesn't attend the computation of metric.

## Write A New Script 

1. copy a existed dataset script in src/datasets, and replace the origin dataset name to your dataset name. 
2. change the num_classes default parameter.
3. override the init function to specific some parameters especially self.file_list, which is a list of Image/Label path correspondence
   + ADE20K  scans **all files** in the image_dir, and replace img_suffix  to seg_map_suffix in the filename.
   + Cityscapes scans files with the given suffix, and get the correspondence in a sorted way.
   + You can refer to the two method above to create your dataset Image/Label correspondence. Both method need to modify the img_suffix and seg_map_suffix  in the correct place.

​	4. In the __init__.py, add your own dataset into if-elif structure in the get_dataset function.

## Create Yaml Config 

1. copy a existed yaml config

2. change the following parameters in Data :DATASET, DATA_PATH，NUM_CLASSES

3. change parameters as you need. To do this, you'd better have a good understanding about the meaning of parameters  in config.

   

Now you can use the command to train on your own dataset.