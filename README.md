# Caffe - dogsvscats

## Preparation

1. Follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.

Execute the command below to enable Python interface of Caffe.
```
export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH
```
2. Get the code and store them under `$CAFFE_ROOT`.
```bash
git clone xxxx
```

`dogsvscats/` is organized as follows.

```
+ dogsvscats/
    + data/
    + scripts/
    + models/
        + caffenet
        + resnet50
```

## Data Preparation

1. Download [
Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/overview) data from Kaggle.  
Unzip the files and store them under `$CAFFE_ROOT/dogsvccats/data/`.  
`data/` should be organized as follows.

```
+ data/
    + train
    + test1
```

2. Create lmdb data and generate the mean file of training data.  
`create_lmdb_meanfile.sh` creates train, val, trainval data in lmdb format and generates meanfile for train and trainval, respectively.  

```bash
bash create_lmdb_meanfile.sh
```

## Train the network

Two networks are provided: CaffeNet and ResNet50. The network definitions and training scripts are stored under `$CAFFE_ROOT/dogsvscats/models/`.

1. Download IMAGENET pretrained models and put them under corresponding directory.
    - [CaffeNet](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel)
    - [ResNet50](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
2. Finetune on dogsvscats data.  

```bash
bash train.sh
or
bash trainval.sh
```

`train.sh` will finetune the pretrained models on train and report accuracy on val. `trainval.sh` will finetune the pretrained models on trainval (the whole training data).

Trained models will be stored under `$CAFFE_ROOT/dogsvscats/moedels/caffnet/exp/`.

## Plot curves

The logs generated during training can be exploited to plot training and testing curves.

```bash
python plot_learning_curve.py
```

This will generate `curve.png` which visualizes the training process.

## Prediction

Finally we can predict on unseen data. Test dataset is provided under `$CAFFE_ROOT/dogsvscats/data/test1`. We can predict on them based on trained models.
```
python make_predictions.py
```

`make_predictions.py` will generate `submission.csv` which can be submitted to Kaggle [
Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/overview) challenge.

