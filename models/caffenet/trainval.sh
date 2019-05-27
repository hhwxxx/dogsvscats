set -e

CAFFE_ROOT="/home/hhw/hhw/work/caffe"
cd $CAFFE_ROOT

export PYTHONATH="$CAFFE_ROOT/python":$PYTHONPATH

./build/tools/caffe train \
    --solver "dogsvscats/models/caffenet/solver_trainval.prototxt" \
    --weights "dogsvscats/models/caffenet/bvlc_reference_caffenet.caffemodel" \
    --gpu 0 2>&1 | tee "dogsvscats/models/caffenet/exp/trainval/caffenet.log"

