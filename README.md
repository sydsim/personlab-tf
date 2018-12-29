# personlab-tf
This is an implemention of [PersonLab](https://arxiv.org/abs/1803.08225) using Tensorflow Slim. Resnet and Mobilenet in slim can be used as a pre-trained model. It is explained in the example notebook to understand how to use it.

# Environment
this project has tested on this environment.

* ubuntu 16.04
* python 3.6
* numpy 1.14.5
* tensorflow 1.10.0
* pycocotools 2.0.0
* scikit-image 0.14.0

# Pretrained Model
[MobileNetV2 Based](https://drive.google.com/file/d/1v_zVLxQSXI69jFIzOZoL0pRWQ1wwoz0f/view?usp=sharing) (Didn't measured the accuracy)

# --- CAUTION ---
Currently, there is an issue that only appears on CPU environment. (https://github.com/sydsim/personlab-tf/issues/7)
