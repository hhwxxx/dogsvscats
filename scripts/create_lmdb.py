from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import glob
import os
import random
import sys

import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb


parser = argparse.ArgumentParser(description='Create lmdb data.')
parser.add_argument('--train_data_path',
                    default='/home/hhw/hhw/work/caffe/dogsvscats/data/train',
                    help='Directory containing the training data.')
parser.add_argument('--test_data_path',
                    default='/home/hhw/hhw/work/caffe/dogsvscats/data/test1',
                    help='Directory containing the testing data.')
parser.add_argument('--save_path',
                    default='/home/hhw/hhw/work/caffe/dogsvscats/data',
                    help='Directory where to save the created lmdb files.')
parser.add_argument('--image_width', type=int, default=227,
                    help='Resize image width.')
parser.add_argument('--image_height', type=int, default=227,
                    help='Resize image height.')
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='trian ratio.')
args = parser.parse_args()


def transform_image(image, image_width, image_height):
    # Histogram Equalization
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])

    # Image Resizing
    image = cv2.resize(image, (image_width, image_height),
                       interpolation=cv2.INTER_CUBIC)

    return image


def make_datum(image, label):
    # image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=args.image_width,
        height=args.image_height,
        label=label,
        data=np.rollaxis(image, 2).tostring())


def create_lmdb(save_path, dataset_split, filelist, start, end):
    lmdb_filename = os.path.join(save_path, dataset_split + '_lmdb')
    if os.path.exists(lmdb_filename):
        os.system('rm -rf ' + lmdb_filename)

    print('Creating {}_lmdb.'.format(dataset_split))

    in_db = lmdb.open(lmdb_filename, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for index in range(start, end):
            filename = filelist[index]
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            image = transform_image(image,
                                    image_width=args.image_width,
                                    image_height=args.image_height)

            if 'cat' in os.path.basename(filename):
                label = 0
            else:
                label = 1

            datum = make_datum(image, label)

            in_txn.put('{:0>5d}'.format(index), datum.SerializeToString())

            ratio = (index - start + 1) / (end - start) * 100
            str_pattern = 'index: {:0>5d}\tfile: {}\tlabel: {}\tratio: {}%'
            print(str_pattern.format(index, os.path.basename(filename),
                                     str(label), ratio))
            
    in_db.close()
    print('Finished creating {}_lmdb.'.format(dataset_split))
    print('{}_lmdb contains {} examples.\n'.format(dataset_split, end-start))


def main():
    random.seed(42)
    train_data = glob.glob(os.path.join(args.train_data_path, '*.jpg'))
    test_data = glob.glob(os.path.join(args.test_data_path, '*.jpg'))
    random.shuffle(train_data)

    trainval_size = len(train_data)
    train_size = int(len(train_data) * args.train_ratio)
    val_size = len(train_data) - train_size

    lmdb = {
        'train': (0, train_size),
        'val': (train_size, trainval_size),
        'trainval': (0, trainval_size),
    }

    for dataset_split, data_range in lmdb.iteritems():
        create_lmdb(args.save_path, dataset_split, train_data, *data_range)


if __name__ == '__main__':
    main()

