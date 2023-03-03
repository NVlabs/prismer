# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import argparse
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import functools
import torch

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from model.prismer_caption import PrismerCaption
from dataset import create_dataset, create_loader
from utils import *
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='')
parser.add_argument('--port', default='')

parser.add_argument('--config', default='configs/pretrain.yaml')
parser.add_argument('--from_checkpoint', action='store_true')
parser.add_argument('--shard_grad_op', action='store_true')
parser.add_argument('--full_shard', action='store_true')
parser.add_argument('--exp_name', default='', type=str)
parser.add_argument('--mixed_precision', default='fp16', type=str)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

train_dataset = create_dataset('pretrain', config)
train_loader = create_loader(train_dataset, batch_size=config['batch_size_train'], num_workers=8, train=True)

model = PrismerCaption(config)
if args.shard_grad_op:  # Model Sharding: ZeRO 2
    from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch, ShardingStrategy, StateDictType, CPUOffload
    fsdp_plugin = FullyShardedDataParallelPlugin(sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                                                 backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                                                 mixed_precision_policy=MixedPrecision(param_dtype=torch.float16,
                                                                                       reduce_dtype=torch.float16,
                                                                                       buffer_dtype=torch.float16),
                                                 state_dict_type=StateDictType.FULL_STATE_DICT,
                                                 cpu_offload=CPUOffload(offload_params=False),
                                                 ignored_modules=model.ignored_modules)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare(model)

elif args.full_shard:  # Model Sharding: ZeRO 3
    from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch, ShardingStrategy, StateDictType
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from model.modules.vit import ResidualAttentionBlock
    from model.modules.resampler import PerceiverAttentionBlock
    from model.modules.roberta import RobertaLayer
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            ResidualAttentionBlock,
            PerceiverAttentionBlock,
            RobertaLayer
        },
    )
    fsdp_plugin = FullyShardedDataParallelPlugin(sharding_strategy=ShardingStrategy.FULL_SHARD,
                                                 backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                                                 mixed_precision_policy=MixedPrecision(param_dtype=torch.float16,
                                                                                       reduce_dtype=torch.float16,
                                                                                       buffer_dtype=torch.float16),
                                                 state_dict_type=StateDictType.FULL_STATE_DICT,
                                                 auto_wrap_policy=auto_wrap_policy,
                                                 ignored_modules=model.ignored_modules)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare(model)
else:
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

# Reload saved states
if args.from_checkpoint:
    state_dict = torch.load(f'logging/pretrain_{args.exp_name}/pytorch_model.bin', map_location='cpu')
    start_epoch = torch.load(f'logging/pretrain_{args.exp_name}/epoch.pt')[0] + 1
    model.load_state_dict(state_dict)
    accelerator.print(f'Start re-training from checkpoint with Epoch {start_epoch}')
else:
    start_epoch = 0

optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),
                              lr=config['init_lr'], weight_decay=config['weight_decay'])

if args.shard_grad_op or args.full_shard:
    optimizer, train_loader = accelerator.prepare(optimizer, train_loader)
else:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)


start_time = time.time()
warmup_step = 0
for epoch in range(start_epoch, config['max_epoch']):
    cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

    train_loss = 0
    num_train_elems = 0
    model.train()
    for i, (experts, caption) in enumerate(tqdm(train_loader)):
        if warmup_step < config['warmup_steps']:
            warmup_lr_schedule(optimizer, warmup_step, config['warmup_steps'], config['warmup_lr'], config['init_lr'])
            warmup_step += 1

        loss = model(experts, caption)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        train_loss += loss.item()
        num_train_elems += 1

    train_loss /= num_train_elems
    accelerator.print(f"Epoch {epoch:03d} | loss: {train_loss:.4f} || Time: {(time.time() - start_time):.4f}")
    accelerator.save_state(f'logging/pretrain_{args.exp_name}')
    accelerator.save([epoch], f'logging/pretrain_{args.exp_name}/epoch.pt')

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
accelerator.print('Training time {}'.format(total_time_str))


