# Object Detection in an Urban Environment

## Project overview
The goal is to detect cars, pedestrians and cyclists in camera input feed.  

## Set up

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).
The data required to finish this project in the workspace, had been already provided. **No download was necessary?!**  

From the instruction in the provided README.md file:  
"The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos.  The `testing` folder contains frames from the 10 fps video without downsampling. You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file."  

In my workspace no `training_and_validation` folder existed. **The folders train, val and test were already there?!**
I nevertheless completed the create_splits.py file as if the setup was according to instructions.  
  
The data for training, validation and testing is organized as shown below(/home/workspace/MY-nd013-c1-vision-starter/data/):
```
    - train: contains the train data (86 files)
    - val: contains the evaluation data (10 files)
    - test - contains 3 files to test model and create inference videos
```
xx

### Experiments

For this project a pretrained SSD Resnet 50 640x640(Single Shot Detector  [here](https://arxiv.org/pdf/1512.02325.pdf)) model was suggested.
The pretrained model had to be downloaded and put into the experiments folder.  
This model is configured by the Tf Object Detection API via config files.  
The provided `pipeline.config` file is the config file for the SSD Resnet 50 640x640 model.
 It has to be adapted for every experiment with the help of also provided `edit_config.py` and `label_map.pbtxt` files. 
 A new file `pipeline_new.config` is produced and has to be placed in the folder of each experiment  
 
The experiments folder(/home/workspace/MY-nd013-c1-vision-starter/experiments/) is organized as shown below:
```
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    
```

## Dataset

### Dataset analysis


### Cross validation


## Training
With the created `pipeline_new.config`  and the provided `model_main_tf2.py` files training and evaluation can be started.

* training:
```
python experiments/model_main_tf2.py --model_dir=experiments/(reference or experimentX)/ --pipeline_config_path=experiments/(reference or experimentX)/pipeline_new.config
```
* evaluation:
```
python experiments/model_main_tf2.py --model_dir=experiments/(reference or experimentX)/ --pipeline_config_path=experiments/(reference or experimentX)/pipeline_new.config --checkpoint_dir=experiments/(reference or experimentX)/
```
The training was monitored with  tensorboard.  

### Reference experiment


### Improve on the reference

The reference experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

