# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os
# import random

import torch
import yaml
from torch.utils.data import DataLoader

# from wenet.dataset.dataset import Dataset
from wenet.utils.config import override_config
# from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.context_graph import ContextGraph
# from wenet.utils.ctc_utils import get_blank_id
# from wenet.bin.train_router import Dataset_pretrain_router, init_dataset_and_dataloader
from wenet.dataset.dataset import Processor, DataList
import wenet.dataset.processor as processor
from wenet.utils.file_utils import read_lists

from wenet.transformer.router import TrainRouterNN
# import torch
# import torchaudio
from wenet.text.base_tokenizer import BaseTokenizer


def Dataset_pretrain_router(data_type,
                            data_list_file,
                            tokenizer: BaseTokenizer,
                            conf,
                            partition=True,
                            percentage=1):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            tokenizer (BaseTokenizer): tokenizer to tokenize
            partition(bool): whether to do data partition in terms of rank
    """
    assert data_type in ['raw', 'shard']
    lists = read_lists(data_list_file)

    n_files = int(len(lists) * percentage)
    # lists = random.choices(lists, k=n_files)
    shuffle = conf.get('shuffle', True)

    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    for idx, batch in enumerate(dataset):
        print('batch: ', batch)
        break
    if data_type == 'shard':
        dataset = Processor(dataset, processor.url_opener)
        dataset = Processor(dataset, processor.tar_file_and_group)
    else:
        dataset = Processor(dataset, processor.parse_raw)

    speaker_conf = conf.get('speaker_conf', None)
    if speaker_conf is not None:
        dataset = Processor(dataset, processor.parse_speaker, **speaker_conf)

    dataset = Processor(dataset, processor.tokenize_router, tokenizer)
    filter_conf = conf.get('filter_conf', {})
    dataset = Processor(dataset, processor.filter, **filter_conf)

    resample_conf = conf.get('resample_conf', {})
    dataset = Processor(dataset, processor.resample, **resample_conf)

    speed_perturb = conf.get('speed_perturb', False)
    if speed_perturb:
        dataset = Processor(dataset, processor.speed_perturb)

    feats_type = conf.get('feats_type', 'fbank')
    assert feats_type in ['fbank', 'mfcc', 'log_mel_spectrogram']
    if feats_type == 'fbank':
        fbank_conf = conf.get('fbank_conf', {})
        dataset = Processor(dataset, processor.compute_fbank, **fbank_conf)
    elif feats_type == 'mfcc':
        mfcc_conf = conf.get('mfcc_conf', {})
        dataset = Processor(dataset, processor.compute_mfcc, **mfcc_conf)
    elif feats_type == 'log_mel_spectrogram':
        log_mel_spectrogram_conf = conf.get('log_mel_spectrogram_conf', {})
        dataset = Processor(dataset, processor.compute_log_mel_spectrogram,
                            **log_mel_spectrogram_conf)

    spec_aug = conf.get('spec_aug', True)
    spec_sub = conf.get('spec_sub', False)
    spec_trim = conf.get('spec_trim', False)
    if spec_aug:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = Processor(dataset, processor.spec_aug, **spec_aug_conf)
    if spec_sub:
        spec_sub_conf = conf.get('spec_sub_conf', {})
        dataset = Processor(dataset, processor.spec_sub, **spec_sub_conf)
    if spec_trim:
        spec_trim_conf = conf.get('spec_trim_conf', {})
        dataset = Processor(dataset, processor.spec_trim, **spec_trim_conf)

    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = Processor(dataset, processor.sort, **sort_conf)

    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)

    dataset = Processor(dataset, processor.padding)

    return dataset


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--blank_penalty',
                        type=float,
                        default=0.0,
                        help='blank penalty')
    parser.add_argument('--result_dir',
                        required=True,
                        help='asr result dir if no result_file')
    parser.add_argument('--result_file',
                        required=False,
                        help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--modes',
                        nargs='+',
                        help="""decoding mode, support the following:
                             attention
                             ctc_greedy_search
                             ctc_prefix_beam_search
                             attention_rescoring
                             rnnt_greedy_search
                             rnnt_beam_search
                             rnnt_beam_attn_rescoring
                             ctc_beam_td_attn_rescoring
                             hlg_onebest
                             hlg_rescore
                             paraformer_greedy_search
                             paraformer_beam_search""")
    parser.add_argument('--search_ctc_weight',
                        type=float,
                        default=1.0,
                        help='ctc weight for nbest generation')
    parser.add_argument('--search_transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for nbest generation')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode')

    parser.add_argument('--transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument('--attn_weight',
                        type=float,
                        default=0.0,
                        help='attention weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    parser.add_argument('--word',
                        default='',
                        type=str,
                        help='word file, only used for hlg decode')
    parser.add_argument('--hlg',
                        default='',
                        type=str,
                        help='hlg file, only used for hlg decode')
    parser.add_argument('--lm_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--r_decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')

    parser.add_argument(
        '--context_bias_mode',
        type=str,
        default='',
        help='''Context bias mode, selectable from the following
                                option: decoding-graph, deep-biasing''')
    parser.add_argument('--context_list_path',
                        type=str,
                        default='',
                        help='Context list path')
    parser.add_argument('--context_graph_score',
                        type=float,
                        default=0.0,
                        help='''The higher the score, the greater the degree of
                                bias using decoding-graph for biasing''')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = True
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size

    tokenizer = init_tokenizer(configs)
    test_dataset = Dataset_pretrain_router(args.data_type,
                                           args.test_data,
                                           tokenizer,
                                           test_conf,
                                           partition=False,
                                           percentage=0.02)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    args.jit = False
    # model, configs = init_model(args, configs)
    model = TrainRouterNN(configs)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()

    context_graph = None
    if 'decoding-graph' in args.context_bias_mode:
        context_graph = ContextGraph(args.context_list_path,
                                     tokenizer.symbol_table,
                                     configs['tokenizer_conf']['bpe_path'],
                                     args.context_graph_score)

    # _, blank_id = get_blank_id(configs, tokenizer.symbol_table)
    # logging.info("blank_id is {}".format(blank_id))

    # TODO(Dinghao Zhou): Support RNN-T related decoding
    # TODO(Lv Xiang): Support k2 related decoding
    # TODO(Kaixun Huang): Support context graph
    # files = {}

    if args.result_file:
        file_name = args.result_file
    else:
        dir_name = args.result_dir
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name, 'text')

    with open(file_name, 'w') as file:
        with torch.no_grad():

            for batch_idx, batch in enumerate(test_data_loader):
                keys = batch["keys"]
                feats = batch["feats"].to(device).squeeze()
                target = batch["target"].to(device)
                feats_lengths = batch["feats_lengths"].to(device)
                target_lengths = batch["target_lengths"].to(device)
                domain = batch['domain'] if 'domain' in batch.keys() else None
                # results = model.decode(
                #     args.modes,
                #     feats,
                #     feats_lengths,
                #     domain,
                #     args.beam_size,
                #     decoding_chunk_size=args.decoding_chunk_size,
                #     num_decoding_left_chunks=args.num_decoding_left_chunks,
                #     ctc_weight=args.ctc_weight,
                #     simulate_streaming=args.simulate_streaming,
                #     reverse_weight=args.reverse_weight,
                #     context_graph=context_graph,
                #     blank_id=blank_id,
                #     blank_penalty=args.blank_penalty)
                result = model.decode(batch, device)

                line = ' '.join(
                    (keys[0], str(batch['domain'].item()), str(result)))
                logging.info(line)
                file.write(line + '\n')

        file.close()
    # for mode, f in files.items():
    #     f.close()


if __name__ == '__main__':
    main()
