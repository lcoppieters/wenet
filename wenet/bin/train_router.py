from __future__ import print_function

import argparse
import datetime
import logging
import os
import torch
import yaml
# import pdb

import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record

from wenet.utils.executor import Executor
from wenet.utils.config import override_config
# from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import (
    add_model_args, add_dataset_args, add_ddp_args, add_deepspeed_args,
    add_trace_args, init_distributed, init_optimizer_and_scheduler,
    trace_and_print_model, wrap_cuda_model, init_summarywriter, save_model,
    log_per_epoch)
from wenet.dataset.dataset import Processor, DataList
import wenet.dataset.processor as processor
from wenet.utils.file_utils import read_lists

import copy

from wenet.transformer.router import TrainRouterNN
# import torch
# import torchaudio
from torch.utils.data import DataLoader
from wenet.text.base_tokenizer import BaseTokenizer


def Dataset_pretrain_router(data_type,
                            data_list_file,
                            tokenizer: BaseTokenizer,
                            conf,
                            partition=True):
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
    # import pdb
    # pdb.set_trace()

    dataset = Processor(dataset, processor.padding)

    return dataset


def init_dataset_and_dataloader(args, configs, tokenizer):
    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['spec_sub'] = False
    cv_conf['spec_trim'] = False
    cv_conf['shuffle'] = False

    # configs['vocab_size'] = tokenizer.vocab_size()
    # import pdb
    # pdb.set_trace()
    train_dataset = Dataset_pretrain_router(args.data_type, args.train_data,
                                            tokenizer, train_conf, True)
    cv_dataset = Dataset_pretrain_router(args.data_type,
                                         args.cv_data,
                                         tokenizer,
                                         cv_conf,
                                         partition=False)

    # NOTE(xcsong): Why we prefer persistent_workers=True ?
    #   https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   persistent_workers=True,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                persistent_workers=True,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--symbol_table', help='path to dict')
    parser.add_argument('--cmvn', help='path to cmvn')
    parser = add_model_args(parser)
    parser = add_dataset_args(parser)
    parser = add_ddp_args(parser)
    parser = add_deepspeed_args(parser)
    parser = add_trace_args(parser)
    args = parser.parse_args()
    if args.train_engine == "deepspeed":
        args.deepspeed = True
        assert args.deepspeed_config is not None
    return args


@record
def main():
    args = get_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Set random seed
    torch.manual_seed(777)
    # Read config
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    if (args.symbol_table is not None) and ('tokenizer_conf'
                                            not in configs.keys()):
        configs['tokenizer_conf'] = {}
        configs['tokenizer_conf']['symbol_table_path'] = args.symbol_table
        configs['tokenizer_conf']['non_lang_syms_path'] = None

    if (args.cmvn is not None) and ('cmvn_conf' not in configs.keys()):
        configs['cmvn_conf'] = {}
        configs['cmvn_conf']['cmvn_file'] = args.cmvn
        configs['cmvn_conf']['is_json_cmvn'] = True
    # init tokenizer
    configs['init_infos'] = {}
    configs['model_dir'] = args.model_dir
    configs['train_engine'] = args.train_engine
    tokenizer = init_tokenizer(configs)

    # Init env for ddp OR deepspeed
    _, _, rank = init_distributed(args)

    # Get dataset & dataloader

    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, tokenizer)
    # model

    saved_config_path = os.path.join(args.model_dir, 'train.yaml')
    with open(saved_config_path, 'w') as fout:
        data = yaml.dump(configs)
        fout.write(data)
    model = TrainRouterNN(configs)
    print(model)
    # Check model is jitable & print model archtectures

    trace_and_print_model(args, model)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # Dispatch model from cpu to gpu
    model, device = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler = init_optimizer_and_scheduler(
        args, configs, model)

    # pdb.set_trace()
    # Save checkpoints
    save_model(model,
               info_dict={
                   "save_time":
                   datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                   "tag":
                   "init",
                   **configs
               })

    # Get executor
    tag = configs["init_infos"].get("tag", "init")
    executor = Executor()
    executor.step = configs["init_infos"].get('step', -1) + int("step_" in tag)

    # Init scaler, used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Start training loop
    start_epoch = configs["init_infos"].get('epoch', 0) + int("epoch_" in tag)
    configs.pop("init_infos", None)
    final_epoch = None

    for epoch in range(start_epoch, configs.get('max_epoch', 100)):
        # import pdb
        # pdb.set_trace()
        train_dataset.set_epoch(epoch)
        configs['epoch'] = epoch

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(
            epoch, lr, rank))

        dist.barrier(
        )  # NOTE(xcsong): Ensure all ranks start Train at the same time.
        # NOTE(xcsong): Why we need a new group? see `train_utils.py::wenet_join`
        group_join = dist.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        with open('save_vals.txt', 'a') as f:
            f.write('Epoch {}'.format(epoch))
        executor.train(model, optimizer, scheduler, train_data_loader,
                       cv_data_loader, writer, configs, scaler, group_join)

        dist.destroy_process_group(group_join)

        dist.barrier(
        )  # NOTE(xcsong): Ensure all ranks start CV at the same time.
        loss_dict = executor.cv(model, cv_data_loader, configs)

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} CV info lr {} cv_loss {} rank {} acc {}'.format(
            epoch, lr, loss_dict["loss"], rank, loss_dict["acc"]))
        info_dict = {
            'epoch': epoch,
            'lr': lr,
            'step': executor.step,
            'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'tag': "epoch_{}".format(epoch),
            'loss_dict': loss_dict,
            **configs
        }

        log_per_epoch(writer, info_dict=info_dict)

        save_model(model, info_dict=info_dict)

        final_epoch = epoch

    if final_epoch is not None and rank == 0:
        final_model_path = os.path.join(args.model_dir, 'final.pt')
        os.remove(final_model_path) if os.path.exists(
            final_model_path) else None
        os.symlink('epoch_{}.pt'.format(final_epoch), final_model_path)
        writer.close()


if __name__ == '__main__':
    main()
