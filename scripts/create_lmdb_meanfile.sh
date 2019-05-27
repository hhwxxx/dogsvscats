set -e

CURRENT_DIR=$(pwd)
CAFFE_ROOT="/home/hhw/hhw/work/caffe"

export PYTHONPATH="$CAFFE_ROOT/python":$PYTHONPATH

# Cretea lmdb files.
python create_lmdb.py \
    --train_data_path "$CAFFE_ROOT/dogsvscats/data/train" \
    --test_data_path "$CAFFE_ROOT/dogsvscats/data/test1" \
    --save_path "$CAFFE_ROOT/dogsvscats/data"

cd $CAFFE_ROOT

# Generate mean file of train data.
./build/tools/compute_image_mean \
    -backend=lmdb \
    "$CAFFE_ROOT/dogsvscats/data/train_lmdb" \
    "$CAFFE_ROOT/dogsvscats/data/mean_train.binaryproto"

# Generate mean file of trainval data.
./build/tools/compute_image_mean \
    -backend=lmdb \
    "$CAFFE_ROOT/dogsvscats/data/trainval_lmdb" \
    "$CAFFE_ROOT/dogsvscats/data/mean_trainval.binaryproto"
