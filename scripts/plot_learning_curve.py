from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import subprocess
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
plt.style.use('ggplot')


parser = argparse.ArgumentParser(description='Plot curves.')
parser.add_argument('--caffe_root',
                    default='/home/hhw/hhw/work/caffe',
                    help='Root directory of caffe.')
parser.add_argument(
    '--log_file',
    default='/home/hhw/hhw/work/caffe/dogsvscats/models/caffenet/exp/train/caffenet.log',
    help='Caffenet training log.')
parser.add_argument(
    '--curve_image',
    default='/home/hhw/hhw/work/caffe/dogsvscats/models/caffenet/exp/train/curve.png',
    help='Caffenet training curve.')

args = parser.parse_args()


def main():
    caffe_root = args.caffe_root
    log_file = args.log_file
    curve_image = args.curve_image

    # Get directory where the model logs is saved, and move to it
    log_dir = os.path.dirname(log_file)
    os.chdir(log_dir)

    # Parsing training/validation logs
    command = os.path.join(caffe_root, 'tools/extra/parse_log.sh') + ' ' + log_file
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

    # Read training and test logs
    train_log_file = log_file + '.train'
    test_log_file = log_file + '.test'
    train_log = pd.read_csv(train_log_file, delim_whitespace=True)
    test_log = pd.read_csv(test_log_file, delim_whitespace=True)


    fig, ax1 = plt.subplots()

    # Plot training and test losses
    train_loss, = ax1.plot(train_log['#Iters'], train_log['TrainingLoss'],
                           color='red', alpha=.5)
    test_loss, = ax1.plot(test_log['#Iters'],test_log['TestLoss'],
                          linewidth=2, color='green')
    ax1.set_ylim(ymin=0, ymax=1)
    ax1.set_xlabel('Iterations', fontsize=15)
    ax1.set_ylabel('Loss', fontsize=15)
    ax1.tick_params(labelsize=15)

    # Plot test accuracy
    ax2 = ax1.twinx()
    test_accuracy, = ax2.plot(test_log['#Iters'], test_log['TestAccuracy'],
                              linewidth=2, color='blue')
    ax2.set_ylim(ymin=0, ymax=1)
    ax2.set_ylabel('Accuracy', fontsize=15)
    ax2.tick_params(labelsize=15)

    # Add legend
    plt.legend([train_loss, test_loss, test_accuracy],
               ['Training Loss', 'Test Loss', 'Test Accuracy'],
               bbox_to_anchor=(1, 0.8))
    plt.title('Training Curve', fontsize=18)

    # Save learning curve
    plt.savefig(curve_image)

    # Delete training and test logs
    command = 'rm ' + train_log_file
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    command = command = 'rm ' + test_log_file
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()


if __name__ == '__main__':
    main()

