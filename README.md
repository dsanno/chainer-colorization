# Chainer implementation of Colorization

[日本語版(Japanese version)](./doc/ja/README.md)

# Referred paper
[Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/ja/)

# Difference from paper

* Use [VGG 16 layers model](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) to extract features of images.
* Not train classification.
* Use Deconvolution2D with stride size 2 for upsampling in colorization network.

# Requirements

* Python 2.7
* [Chainer 1.8.0](http://chainer.org/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)

# Usage

## Locate training images

Put training images into one directory.

## Download VGG 16 layers model

Download VGG_ILSVRC_16_layers.caffemodel from [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) and put it into the root directory.

## Convert VGG 16 layers model

It is slow to load a Caffe model with Chainer, so convert VGG 16 layers Caffe model to faster format and remove unused parameters.

```
$ python src/create_chainer_model.py
```

## Convert training images

Convert training images to pkl file.

```
$ python src/convert_dataset.py <training images dir> <output pkl file path> -n <maximum number of image files>
```

Example:

```
$ python src/convert_dataset.py dataset/image dataset/images.pkl -n 200000
```

## Train model

Example:

```
$ python src/train.py -g 0 -m vgg16.model -o model/color --out_image_dir image -d dataset/images.pkl --batch_size 48
```

Options:

* -g (--gpu) <GPU device index>: optional  
GPU device index. Negative number indicates using CPU (default: -1)
* -m (--model) <VGG 16 layers model filea path>: optional  
Converted VGG 16 layers model file path (default: vgg16.model)
* -i (--input) <model file path>: optional  
Input model file path without extension
* -o (--output) <model file path>: required  
Output model file path without extension  
Saved model file names are:
    * model parameter file: <model file path>_<iteration number>.model
    * optimizer paremeter file: <model file path>_<iteration number>.state
* -d (--dataset) <dataset file path>: optional  
Dataset file path (default: dataset/images.pkl)
* --iter <number of iteration>: optional  
The number of iteration (default: 100)
* --batch_size <mini batch size>: optional  
Mini batch size(default: 48)
* --out_image_dir <image output directory path>: optional  
Directory path to output colorized images while training  
Images are not output if not set

# License

MIT
