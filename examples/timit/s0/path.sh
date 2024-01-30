export WENET_DIR=$PWD/../../..
export BUILD_DIR=${WENET_DIR}/runtime/libtorch/build
export OPENFST_PREFIX_DIR=${BUILD_DIR}/../fc_base/openfst-subbuild/openfst-populate-prefix
export PATH=$PWD:${BUILD_DIR}/bin:${BUILD_DIR}/kaldi:${OPENFST_PREFIX_DIR}/bin:$PATH:/idiap/home/lcoppieters/.local/bin

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=/idiap/user/lcoppieters/anaconda3/envs/fairseq2022/bin/python3:/usr/bin/python3:$PYTHONPATH
