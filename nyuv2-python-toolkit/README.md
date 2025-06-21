# NYUv2 Python Toolkit

<div>
<img src="test/image.png"        width="18%">
<img src="test/semantic13.png"   width="18%">
<img src="test/semantic40.png"   width="18%">
<img src="test/depth.png"        width="18%">
<img src="test/normal.png"       width="18%">
</div>

This repo provides extraction tools and a pytorch dataloader for NYUv2 dataset. All meta data comes from [ankurhanda/nyuv2-meta-data](https://github.com/ankurhanda/nyuv2-meta-data) 

Supported Tasks:

* Semantic Segmentation (13 classes and 40 classes)
* Depth Estimation
* Normal

## Extraction

```bash
bash download_and_extract.sh
```
or 

```bash
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
wget https://inf.ethz.ch/personal/ladickyl/nyu_normals_gt.zip

python extract_nyuv2.py --mat nyu_depth_v2_labeled.mat --normal_zip nyu_normals_gt.zip  --data_root NYUv2 --save_colored
```
All images and labels will be extracted to ./NYUv2 as the following:

```
NYUv2/
    image/
        train/
            00003.png
            00004.png
            ...
        test/
    seg13/
        train/
            00003.png
            00004.png
            ...
        test/
    seg40/
    depth/
    normal/
```


# nyuv2-meta-data

## What does this repository contain?

This repository contains 13 class labels for both train and test dataset in NYUv2. This is to avoid any hassle involved in parsing the data from the .mat files. If you are looking to train a network to do 13 class segmentation from RGB data, then this repository can provide you both the training/test dataset as well the corresponding ground truth labels. However, if your networks needs additionally depth data (either depth image or DHA features) then you will need to download the dataset from the [NYUv2 website](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) (~2.8GB) as well as the corresponding [toolbox](http://cs.nyu.edu/~silberman/code/toolbox_nyu_depth_v2.zip). To summarise, this repository contains the following

- The **train_labels_13** contains the ground truth annotation for 13 classes for NYUv2 training dataset while **test_labels_13** contains the ground truth for test dataset in NYUv2.

- The training dataset (795 RGB images) can be obtained from [nyu_train_rgb](http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz) (277MB) while the test dataset (654 RGB images) can be obtained from [nyu_test_rgb](http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz) (227MB).

- Important to remember that the label files are ordered but the rgb files are not. Though you can order the files using ``gprename``.

## How do I obtain the DHA features?

Look for this in a corresponding [SUN RGB-D meta data repository](https://github.com/ankurhanda/sunrgbd-meta-data). You will need rotation matrices for each training and test image. They are available here at [**camera_rotations_NYU.txt**](https://github.com/ankurhanda/nyuv2-meta-data/blob/master/camera_rotations_NYU.txt). These matrices are used to align the floor normal vector to the canonical gravity vector. There are 1449 rotation matrices in total and the indices for these matrices corresponding to training and test data are in **splits.mat**. Remember that labels are ordered *i.e.* training labels files are named with indices 1 to 795 and similarly for test dataset. 

## How do I benchmark? 
[getAccuracyNYU.m](https://github.com/ankurhanda/SceneNetv1.0/blob/master/getAccuracyNYU.m) available in the [SceneNetv1.0](https://github.com/ankurhanda/SceneNetv1.0/) repository allows you to obtain the avereage global and class accuracies. 

## What are the classes and where is the mapping form the class number to the class name?

The mapping is also available at [SceneNetv1.0](https://github.com/ankurhanda/SceneNetv1.0/) repository.








