#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage="1-1"
space="<space>"
track="isolated_1ch_track"
track="isolated"
wsj1_data_dir=/idiap/resource/database/WSJ1
chime4_data_dir=/idiap/resource/database/CHiME4
dump_wav_dir=/idiap/temp/lcoppieters/tmp_chime4/wav_wsj0

data_dir=data/chime4
dict=$data_dir/dict_char.txt
train_config=conf/train_conformer.yaml
exp_dir=/idiap/temp/lcoppieters/tmp_chime4/exp/ch1_and_wsj0
decode_modes="ctc_prefix_beam_search attention_rescoring"
average_checkpoint=true
average_num=10
checkpoint=
. ./tools/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

echo $beg $end
if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: preparing data ..."
  ./local/clean_wsj0_data_prep.sh $chime4_data_dir/CHiME3/data/WSJ0
  # echo "----------------- STEP1 DONE -------------------"
  # ./local/simu_noisy_chime4_data_prep.sh $chime4_data_dir/CHiME3
  # echo "----------------- STEP2 DONE -------------------"
  # ./local/real_noisy_chime4_data_prep.sh $chime4_data_dir/CHiME3
  # echo "----------------- STEP3 DONE -------------------"
  ./local/simu_enhan_chime4_data_prep.sh ${track}_CH1 $chime4_data_dir/CHiME3/data/audio/16kHz/$track
  echo "----------------- STEP4 DONE -------------------"
  echo $track $chime4_data_dir/CHiME3/data/audio/16kHz/$track
  ./local/real_enhan_chime4_data_prep.sh ${track}_CH1 $chime4_data_dir/CHiME3/data/audio/16kHz/$track
  echo "----------------- STEP5 DONE -------------------"
  # ./local/clean_wsj1_data_prep.sh $wsj1_data_dir/media
  # echo "----------------- STEP6 DONE -------------------"
  ./local/chime4_format_dir_ch1_and_wsj0.sh
  echo "----------------- STEP7 DONE -------------------"

fi


if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo -e "<NOISE>\n<*IN*>\n<*MR.*>" > $data_dir/train_wsj1/non_lang.txt
  for name in dev train; do
    python3 tools/text2token.py $data_dir/$name/text -n 1 -s 1 \
      -l $data_dir/train/non_lang.txt > $data_dir/$name/char
  done
  mkdir -p $(dirname $dict) && echo -e "<blank> 0\n<unk> 1" > ${dict}
  cat $data_dir/train/char | cut -f 2- -d" " | tr " " "\n" | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict
  echo "Make dictionary done"
fi


if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  # ./local/chime4_gen_wav.sh $data_dir/train ${dump_wav_dir}_train
  # ./local/chime4_gen_wav.sh $data_dir/dev ${dump_wav_dir}_dev
  # ./local/chime4_gen_wav.sh $data_dir/test ${dump_wav_dir}_test
  # tools/compute_cmvn_stats.py --num_workers 16 \
  #  --train_config $train_config \
  #  --in_scp $data_dir/train/wav.scp \
  #  --out_cmvn $data_dir/train/global_cmvn
  # echo "Prepare data, prepare required format"
  for x in train dev test; do
    tools/make_raw_list.py $data_dir/$x/wav.scp $data_dir/$x/text \
    $data_dir/$x/data.list
  done
fi

if [ $end -ge 4 ] && [ $beg -le 4 ]; then
  echo start stage 4
  mkdir -p $exp_dir && cp $data_dir/train/global_cmvn $exp_dir
  echo $train_config 
  echo $data_dir/train/data.list 
  echo $data_dir/dev/data.list 
  echo $exp_dir 
  echo $dict 
  echo $exp_dir/global_cmvn 
  echo $exp_dir/train.log 
  # python3 wenet/bin/train.py \
  
  python3 -m torch.distributed.run wenet/bin/train.py \
    --config $train_config \
    --train_data $data_dir/train/data.list \
    --cv_data $data_dir/dev/data.list \
    --model_dir $exp_dir \
    --num_workers 4 \
    --symbol_table $dict \
    --cmvn $exp_dir/global_cmvn \
    --pin_memory > $exp_dir/train.log 2>&1
    # --checkpoint $checkpoint \
  echo end stage 4
fi

suffix="isolated_1ch_track"
if [ $end -ge 5 ] && [ $beg -le 5 ]; then
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$exp_dir/avg_${average_num}.pt
  #   echo "do model average and final checkpoint is $decode_checkpoint"
  #   python wenet/bin/average_model.py \
  #     --dst_model $decode_checkpoint \
  #     --src_path $exp_dir  \
  #     --num ${average_num} \
  #     --val_best
  fi
  nj=4
  ctc_weight=0.5
  echo 'before first for loop' 
  for x in dt05_{simu,real} et05_{simu,real}; do
    subdir=${x}_${suffix}
    tools/make_raw_list.py $data_dir/$subdir/wav.scp $data_dir/$subdir/text \
      $data_dir/$subdir/data.list
  done
  echo 'before second for loop' 
  for mode in ${decode_modes}; do
    echo mode: $mode
    # for x in dt05_{simu,real} et05_{simu,real}; do
    for x in dt05_simu; do
      # subdir=${x}_${suffix}
      subdir=test
      dec_dir=$exp_dir/${subdir}_${mode} && mkdir -p $dec_dir
      echo $dec_dir/text
      # python3 wenet/bin/recognize.py \
      #   --gpu 0 \
      #   --mode $mode \
      #   --config $exp_dir/train.yaml \
      #   --test_data $data_dir/$subdir/data.list \
      #   --checkpoint $decode_checkpoint \
      #   --beam_size 8 \
      #   --batch_size 1 \
      #   --result_dir $exp_dir \
      #   --ctc_weight $ctc_weight #ß > $dec_dir/text
        
        # $exp_dir/final_2.pt \
        # --result_file $dec_dir/text \
        # --dict $dict
        # --dict $--checkpoint $exp_dir/avg_${average_num}.pt \dict \--result_file $dec_dir/text &
     done
    #  wait
  done
  echo 'before third for loop' 

  for mode in ${decode_modes}; do
    # for x in dt05_{simu,real} et05_{simu,real}; do
    for x in dt05_simu; do
     subdir=${x}_${suffix}
     subdir=test
     dec_dir=$exp_dir/${mode}
     echo $data_dir/$subdir/text
     echo $dec_dir/text
     echo $dec_dir/wer
     sed -i 's:<unk>: :g' $dec_dir/text
     python3 tools/compute-wer.py --char=1 --v=1 \
       $data_dir/$subdir/text $dec_dir/text  > $dec_dir/wer
    done
  done
fi

