# Performance Record

## Conformer Result

* Feature info: dither + specaug + speed perturb
* Training info: lr 0.0005, batch size 8, 1 gpu, acc_grad 4, 80 epochs
* Decoding info: average_num 10

|      decoding mode     | dt05_real_1ch | dt05_simu_1ch | et05_real_1ch | et05_simu_1ch |
|:----------------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| ctc_prefix_beam_search |   19.06%      |   21.17%      |   28.39%      |    29.16%     |
|  attention_rescoring   |   17.92%      |   20.22%      |   27.40%      |    28.25%     |


## Code adaptations:

* stage 1: data preparation

The idiap data set was a simple link to CHiME3, the isolated_1ch_track didn't contain the training wav-files, but only the dev and training set

The directory $chime4_data_dir should become $chime4_data_dir/CHiME3

Replace $wsj1_data_dir by $wsj1_data_dir/media

To select the CH1 manually: 
line 10: track=track="isolated"

line 35: ./local/simu_enhan_chime4_data_prep.sh ${track}_CH1 $chime4_data_dir/CHiME3/data/audio/16kHz/$track
line 36: ./local/real_enhan_chime4_data_prep.sh ${track}_CH1 $chime4_data_dir/CHiME3/data/audio/16kHz/$track

in the files /local/simu_enhan_chime4_data_prep.sh and /local/real_enhan_chime4_data_prep.sh: 
replace all: "_REAL" by ".CH1_REAL" to match the right channel

you can also choose another channel

To chose which files you want to use as train, dev and test set, you can adapt /local/chime4_format_dir.sh


* stage 2 :

nothing special, except the fact that you should be aware that the tokenization also happens in the training loop, using the 

* stage 3: get cmvn and data.list files

The computation of the wav files is commented in the baseline implementation:
in ./local/chime4_gen_wav.sh you should uncomment lines 17-18.
This process takes a long time: it converts the wv1 files in wav files with sph2pipe.

To verify everything went well:
check if the following numbers are equal:
Orginal utterances (.wav + .wv1): ...
Wave utterances (.wav): ...

If not, check then if the sph_wav contains the wsj0 and wsj1 data and if raw_wav contains the noisy datasets.

If one of those files is empty, something went wrong and since the original wav-file has been overwritten, you should fix the bug and then rerun stage 1 to make it work. 

* stage 4: 
The variables  --symbol_table $dict and --cmvn $exp_dir/global_cmvn are not available in wenet/bin/train.py 

To solve this issue, you can add in wenet/bin/train.py in def args():
    parser.add_argument('--symbol_table', help='path to dict')
    parser.add_argument('--cmvn', help='path to cmvn')

Then in main():
    if args.symbol_table is not None:
        configs['tokenizer_conf']={}
        configs['tokenizer_conf']['symbol_table_path'] = args.symbol_table
        configs['tokenizer_conf']['non_lang_syms_path'] = None

TODO: fix the cmvn computation

* stage 5:
    in the arguments of wenet/bin/recognize.py
    --result_file
    --dict



