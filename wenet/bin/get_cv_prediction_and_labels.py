# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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
import datetime
import logging
import os
import torch
import yaml

# import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record

from wenet.utils.executor import Executor
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import (
    add_model_args, add_dataset_args, add_ddp_args, add_deepspeed_args,
    add_trace_args, init_dataset_and_dataloader, check_modify_and_save_config,
    init_optimizer_and_scheduler, trace_and_print_model, init_summarywriter,
    save_model, log_per_epoch)


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


# NOTE(xcsong): On worker errors, this recod tool will summarize the
#   details of the error (e.g. time, rank, host, pid, traceback, etc).
@record
def main():
    # import pdb
    # pdb.set_trace()
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

    tokenizer = init_tokenizer(configs)

    # Init env for ddp OR deepspeed
    # _, _, rank = init_distributed(args)
    rank = 0

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, tokenizer)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs,
                                           tokenizer.symbol_table)

    # Init asr model from configs
    model, configs = init_model(args, configs)

    # Check model is jitable & print model archtectures
    trace_and_print_model(args, model)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # Dispatch model from cpu to gpu
    # model, device = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler = init_optimizer_and_scheduler(
        args, configs, model)

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
    # if args.use_amp:
    #     scaler = torch.cuda.amp.GradScaler()

    # Start training loop
    start_epoch = configs["init_infos"].get('epoch', 0) + int("epoch_" in tag)
    configs.pop("init_infos", None)
    final_epoch = None
    epoch = start_epoch
    # for epoch in range(start_epoch, configs.get('max_epoch', 100)):
    # train_dataset.set_epoch(epoch)
    configs['epoch'] = epoch

    lr = optimizer.param_groups[0]['lr']
    logging.info('Epoch {} TRAIN info lr {} rank {}'.format(epoch, lr, rank))

    print('here ok before cv')

    # dist.barrier(
    #     )  # NOTE(xcsong): Ensure all ranks start Train at the same time.
    # NOTE(xcsong): Why we need a new group? see `train_utils.py::wenet_join`
    # group_join = dist.new_group(
    #     backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
    # import pdb
    # pdb.set_trace()
    # executor.train(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, configs, scaler, group_join)

    # dist.destroy_process_group(group_join)
    print('here ok before cv')

    # dist.barrier(
    # )  # NOTE(xcsong): Ensure all ranks start CV at the same time.
    loss_dict = executor.cv(model, cv_data_loader, configs)
    print('here ok after cv')

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