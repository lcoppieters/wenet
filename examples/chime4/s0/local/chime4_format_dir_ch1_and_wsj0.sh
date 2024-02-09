#!/usr/bin/env bash

# wujian@2020

set -eu

echo "$0: Formating chime4 data dir..."

track=isolated_CH1
data_dir=data/chime4

mkdir -p $data_dir/{train,dev,test}

cat $data_dir/tr05_{simu,real}_${track}/wav.scp $data_dir/tr05_orig_clean/wav.scp | sort -k1 > $data_dir/train/wav.scp
cat $data_dir/tr05_{simu,real}_${track}/text $data_dir/tr05_orig_clean/text | sort -k1 > $data_dir/train/text

cat $data_dir/dt05_{real,simu}_${track}/wav.scp $data_dir/dt05_orig_clean/wav.scp | sort -k1 > $data_dir/dev/wav.scp
cat $data_dir/dt05_{real,simu}_${track}/text $data_dir/dt05_orig_clean/wav.scp | sort -k1 > $data_dir/dev/text


cat $data_dir/et05_{real,simu}_${track}/wav.scp $data_dir/et05_orig_clean/wav.scp | sort -k1 > $data_dir/test/wav.scp
cat $data_dir/et05_{real,simu}_${track}/text $data_dir/et05_orig_clean/wav.scp | sort -k1 > $data_dir/test/text

echo "$0: Format $data_dir done"
