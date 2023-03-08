# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import argparse
import numpy as np
import random
import time
import functools
import json
import torch
import os
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
    
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from model.prismer_caption import PrismerCaption
from model.modules.utils import interpolate_pos_embed
from dataset import create_dataset, create_loader
from utils import *
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='')
parser.add_argument('--port', default='')

parser.add_argument('--config', default='configs/caption.yaml')
parser.add_argument('--from_checkpoint', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--target_dataset', default='coco', type=str)
parser.add_argument('--shard_grad_op', action='store_true')
parser.add_argument('--full_shard', action='store_true')
parser.add_argument('--exp_name', default='', type=str)
parser.add_argument('--mixed_precision', default='fp16', type=str)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)[args.target_dataset]
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

train_dataset, test_dataset = create_dataset('caption', config)
train_loader = create_loader(train_dataset, batch_size=config['batch_size_train'], num_workers=8, train=True)
test_loader = create_loader(test_dataset, batch_size=config['batch_size_test'], num_workers=8, train=False)


model = PrismerCaption(config)
tokenizer = model.tokenizer

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
    state_dict = torch.load(f'logging/caption_{args.exp_name}/pytorch_model.bin', map_location='cpu')
    if os.path.exists(f'logging/caption_{args.exp_name}/epoch.pt'):
        start_epoch = torch.load(f'logging/caption_{args.exp_name}/epoch.pt')[0] + 1
    else:
        start_epoch = 0
    model.load_state_dict(state_dict)
    accelerator.print(f'Start re-training from checkpoint with Epoch {start_epoch}')

optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()),
                              lr=config['init_lr'], weight_decay=config['weight_decay'])

if args.shard_grad_op or args.full_shard:
    optimizer, train_loader, test_loader = accelerator.prepare(optimizer, train_loader, test_loader)
else:
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

best = 0
start_time = time.time()
if not args.evaluate:
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

        model.eval()
        result = []
        with torch.no_grad():
            for step, (experts, data_ids) in enumerate(tqdm(test_loader)):
                captions = model(experts, train=False, prefix=config['prefix'])

                if accelerator.use_distributed:
                    captions = tokenizer(captions, max_length=30, padding='max_length', return_tensors='pt').input_ids
                    captions = captions.to(experts['rgb'].device)
                    data_ids, captions = accelerator.gather_for_metrics((data_ids, captions))

                    for data_id, caption in zip(data_ids, captions):
                        caption = tokenizer.decode(caption, skip_special_tokens=True)
                        if args.target_dataset == 'coco':
                            image_id = int(test_loader.dataset.data_list[data_id]['image'].split('/')[-1].strip('.jpg').split('_')[-1])
                            result.append({"image_id": image_id, "caption": caption.capitalize() + '.'})
                        elif args.target_dataset == 'nocaps':
                            result.append({"image_id": test_loader.dataset.data_list[data_id]['img_id'],
                                           "caption": caption.capitalize() + '.'})

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            json.dump(result, open(f'/results/caption_results_{args.exp_name}_{args.target_dataset}.json', 'w'))
            if args.target_dataset == 'coco':
                coco_eval = coco_caption_eval(f'{config["data_path"]}/coco_karpathy_test_gt.json', result)
                torch.save([coco_eval.eval['CIDEr']], f'logging/caption_{args.exp_name}/temp_cider.pt')
                if not os.path.isfile(f'logging/caption_{args.exp_name}/cider.pt'):
                    torch.save([coco_eval.eval['CIDEr']], f'logging/caption_{args.exp_name}/cider.pt')

        accelerator.wait_for_everyone()
        cider = torch.load(f'logging/caption_{args.exp_name}/cider.pt')[0]
        curr_cider = torch.load(f'logging/caption_{args.exp_name}/temp_cider.pt')[0]

        if cider < curr_cider:
            train_loss /= num_train_elems
            accelerator.print(f"Epoch {epoch:03d} | loss: {train_loss:.4f} || Time: {(time.time() - start_time):.4f}")
            accelerator.save_state(f'logging/caption_{args.exp_name}')
            accelerator.save([epoch], f'logging/caption_{args.exp_name}/epoch.pt')
            accelerator.save([curr_cider], f'logging/caption_{args.exp_name}/cider.pt')


model.eval()
if accelerator.is_main_process:
    result = []

with torch.no_grad():
    for step, (experts, data_ids) in enumerate(tqdm(test_loader)):
        captions = model(experts, train=False, prefix=config['prefix'])

        if accelerator.use_distributed:
            captions = tokenizer(captions, max_length=30, padding='max_length', return_tensors='pt').input_ids
            captions = captions.to(experts['rgb'].device)
            data_ids, captions = accelerator.gather_for_metrics((data_ids, captions))

        if accelerator.is_main_process:
            for data_id, caption in zip(data_ids, captions):
                caption = tokenizer.decode(caption, skip_special_tokens=True)
                if args.target_dataset == 'coco':
                    image_id = int(test_loader.dataset.data_list[data_id]['image'].split('/')[-1].strip('.jpg').split('_')[-1])
                    result.append({"image_id": image_id, "caption": caption.capitalize() + '.'})
                elif args.target_dataset == 'nocaps':
                    result.append({"image_id": test_loader.dataset.data_list[data_id]['img_id'],
                                   "caption": caption.capitalize() + '.'})

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    json.dump(result, open(f'/results/caption_results_{args.exp_name}_{args.target_dataset}.json', 'w'))
    if args.target_dataset == 'coco':
        coco_caption_eval(f'{config["data_path"]}/coco_karpathy_test_gt.json', result)


