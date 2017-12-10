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
``