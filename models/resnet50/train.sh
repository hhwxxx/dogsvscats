set -e

CAFFE_ROOT="/home/hhw/hhw/work/caffe"
cd $CAFFE_ROOT

export PYTHONATH="$CAFFE_ROOT/python":$PYTHONPATH

./build/tools/caffe train \
    --solver "dogsvscats/models/resnet50/solver_train.prototxt" \
    --weights "dogsvscats/models/resnet50/ResNet-50-model.caffemodel" \
    --gpu 0 2>&1 | tee "dogsvscats/models/resnet50/exp/train/resnet50.log"

