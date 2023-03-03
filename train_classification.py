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
import functools
import torch

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from model.prismer_caption import PrismerCaption
from model.modules.utils import interpolate_pos_embed
from dataset import create_dataset, create_loader
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='')
parser.add_argument('--port', default='')

parser.add_argument('--config', default='configs/classification.yaml')
parser.add_argument('--from_checkpoint', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--exp_name', default='', type=str)
parser.add_argument('--shard_grad_op', action='store_true')
parser.add_argument('--full_shard', action='store_true')
parser.add_argument('--mixed_precision', default='fp16', type=str)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

train_dataset, test_dataset = create_dataset('classification', config)
train_loader = create_loader(train_dataset, batch_size=config['batch_size_train'], num_workers=8, train=True)
test_loader = create_loader(test_dataset, batch_size=config['batch_size_test'], num_workers=8, train=False)
model = PrismerCaption(config)

if args.shard_grad_op:  # Model Sharding: ZeRO 2
    from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch, ShardingStrategy, StateDictType
    fsdp_plugin = FullyShardedDataParallelPlugin(sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                                                 backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                                                 mixed_precision_policy=MixedPrecision(param_dtype=torch.float16,
                                                                                       reduce_dtype=torch.float16,
                                                                                       buffer_dtype=torch.float16),
                                                 state_dict_type=StateDictType.FULL_STATE_DICT,
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
if not args.from_checkpoint:
    state_dict = torch.load(f'logging/pretrain_{args.exp_name}/pytorch_model.bin', map_location='cpu')
    state_dict['expert_encoder.positional_embedding'] = interpolate_pos_embed(state_dict['expert_encoder.positional_embedding'],
                                                                              len(model.expert_encoder.positional_embedding))
    model.load_state_dict(state_dict)
    start_epoch = 0
else:
    state_dict = torch.load(f'logging/classification_{args.exp_name}/pytorch_model.bin', map_location='cpu')
    start_epoch = torch.load(f'logging/classification_{args.exp_name}/epoch.pt')[0] + 1
    model.load_state_dict(state_dict)
    accelerator.print(f'Start re-training from checkpoint with Epoch {start_epoch}')

optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),
                              lr=config['init_lr'], weight_decay=config['weight_decay'])

if args.shard_grad_op or args.full_shard:
    optimizer, train_loader, test_loader = accelerator.prepare(optimizer, train_loader, test_loader)
else:
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

start_time = time.time()
best = 0
for epoch in range(start_epoch, config['max_epoch']):
    train_loss = 0
    num_train_elems = 0
    model.train()
    for i, (experts, caption) in enumerate(tqdm(train_loader)):
        cosine_lr_schedule(optimizer, epoch * len(train_loader) + i, config['max_epoch'] * len(train_loader), config['init_lr'], config['min_lr'])
        loss = model(experts, caption, prefix=config['prefix'])

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        train_loss += loss.item()
        num_train_elems += 1
        
    train_loss /= num_train_elems
    accelerator.print(f"Epoch {epoch:03d} | loss: {train_loss:.4f} || Time: {(time.time() - start_time):.4f}")
    
    if (epoch + 1) % 5 == 0:
        model.eval()
        num_test_elems = 0
        accurate = 0
        with torch.no_grad():
            answer_list = test_loader.dataset.answer_list
            for step, (experts, gt) in enumerate(tqdm(test_loader)):
                predictions = model(experts, answer=answer_list, train=False, prefix=config['prefix'], k_test=config['k_test'], inference='rank')
    
                if accelerator.use_distributed:
                    predictions, gt = accelerator.gather_for_metrics((predictions, gt))

                accurate_preds = predictions == gt
                num_test_elems += accurate_preds.shape[0]
                accurate += accurate_preds.long().sum()
            eval_metric = accurate.item() / num_test_elems
       
        accelerator.wait_for_everyone()
        accelerator.print(f'{config["shots"]}-Shot Acc: {eval_metric}')
    
        if eval_metric > best:
            best = eval_metric
            accelerator.save_state(f'logging/classification_{args.exp_name}')
            accelerator.save([epoch], f'logging/classification_{args.exp_name}/epoch.pt')









