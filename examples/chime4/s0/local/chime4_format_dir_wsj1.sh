#!/usr/bin/env bash

# wujian@2020

set -eu

echo "$0: Formating chime4 data dir..."

track=isolated_1ch_track
data_dir=data/chime4

mkdir -p $data_dir/{train_wsj1,dev_wsj1}

cat $data_dir/train_si200_wsj1_clean/wav.scp | sort -k1 > $data_dir/train_wsj1/wav.scp
cat $data_dir/train_si200_wsj1_clean/text | sort -k1 > $data_dir/train_wsj1/text

cat $data_dir/dt05_real_${track}/wav.scp | sort -k1 > $data_dir/dev_wsj1/wav.scp
cat $data_dir/dt05_real_${track}/text | sort -k1 > $data_dir/dev_wsj1/text

echo "$0: Format $data_dir done"
