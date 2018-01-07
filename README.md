# dl_VQA
Deep Learning project about Visual Question Answering using TensorFLow


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Python 2.7 and Tensorflow have to be installed

### Download Data and Preprocess

In the /data directory, execute the following to download the data and save it into raw json files

```bash
python vqa_preprocessing.py --download True --split 1
python vqa_preprocessing.py --split 2
python vqa_preprocessing.py --split 3
```

Back in the main directory, process the raw data into question+answer+vocab files
```bash
python preprocess.py --split 1 --subset False --num_ans 1000
python preprocess.py --split 2 --subset False --num_ans 1000
python preprocess.py --split 3 --subset False --num_ans 1000
```

### Split Images into 9 overlapping sub-images

preprocess_img.py expects the train-/ test-/ val- data to be in the directory "/train2014" / "/test2014" / "/val2014". To split the images execute the following command
```bash
python preprocess_img.py
```
This will save the sub-images in the following directories "/sub_img_train2014", "/sub_img_test2014" and "/sub_img_val2014"

### Get Features using Resnet

You will need to checkout the Tensorflow models repository. To do so, execute
```bash
git clone https://github.com/tensorflow/models/
```

To finish the setup you will need to add the directory <checkout_dir>/research/slim to your $PYTHONPATH variable.

We are now ready to extract the features of the sub-images. To do so, execute

```bash
example_feat_extract.py 
--network resnet_v1_101 
--checkpoint ./checkpoints/resnet_v1_101.ckpt 
--image_path ./images_dir/ 
--out_file ./features.h5
--num_classes 1000 
--layer_names resnet_v1_101/global_pool
```