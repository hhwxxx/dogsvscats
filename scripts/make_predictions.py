from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import os
import glob

import caffe
from caffe.proto import caffe_pb2
import cv2
import lmdb
import numpy as np

CAFFE_ROOT = '/home/hhw/hhw/work/caffe'

parser = argparse.ArgumentParser(description='Make predictions.')
parser.add_argument('--image_width', type=int, default=227,
                    help='Resize image width.')
parser.add_argument('--image_height', type=int, default=227,
                    help='Resize image height.')
parser.add_argument('--mean_file',
                    default='/home/hhw/hhw/work/caffe/dogsvscats/data/mean_train.binaryproto',
                    help='Mean file of training data.')
parser.add_argument(
    '--deploy_prototxt',
    default='/home/hhw/hhw/work/caffe/dogsvscats/models/caffenet/caffenet_deploy.prototxt',
    help='Deploy prototxt.')
parser.add_argument(
    '--caffemodel',
    default='/home/hhw/hhw/work/caffe/dogsvscats/models/caffenet/exp/train/caffenet_iter_40000.caffemodel',
    help='Trained caffe model.')
parser.add_argument('--test_data_path',
                    default='/home/hhw/hhw/work/caffe/dogsvscats/data/test1',
                    help='Directory containing test data.')
parser.add_argument('--csv_file',
                    default='/home/hhw/hhw/work/caffe/dogsvscats/caffenet/exp/submission.csv',
                    help='Submission csv filename.')

args = parser.parse_args()


def transform_image(image, image_width, image_height):
    """Image processing helper function."""
    # Histogram Equalization
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])

    # Image Resizing
    image = cv2.resize(image,
                       (image_width, image_height),
                       interpolation=cv2.INTER_CUBIC)

    return image


def get_mean_file(mean_file):
    """Read mean blob."""
    mean_blob = caffe_pb2.BlobProto()
    with open(mean_file) as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(
        mean_blob.data,
        dtype=np.float32).reshape((mean_blob.channels,
                                   mean_blob.height,
                                   mean_blob.width))

    return mean_array


def main():
    # Read model architecture and trained model's weights
    net = caffe.Net(args.deploy_prototxt, args.caffemodel, caffe.TEST)

    # Define image transformers
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    mean_array = get_mean_file(args.mean_file)
    transformer.set_mean('data', mean_array)
    transformer.set_transpose('data', (2,0,1))

    # Reading image files
    test_images = glob.glob(os.path.join(args.test_data_path, '*.jpg'))

    # Making predictions
    test_ids = []
    preds = []
    for image_filename in test_images:
        image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
        image = transform_image(image,
                                image_width=args.image_width,
                                image_height=args.image_height)
        
        net.blobs['data'].data[...] = transformer.preprocess('data', image)
        out = net.forward()
        pred_probas = out['prob']

        test_ids = test_ids + [image_filename.split('/')[-1][:-4]]
        preds = preds + [pred_probas.argmax()]

        print(os.path.basename(image_filename), end='\t')
        print(pred_probas.argmax())
        print('-' * 50)

    # Making submission file
    with open(args.csv_file, "w") as f:
        f.write("id,label\n")
        for i in range(len(test_ids)):
            f.write(str(test_ids[i]) + "," + str(preds[i]) + "\n")


if __name__ == '__main__':
    caffe.set_mode_gpu() 
    main()

