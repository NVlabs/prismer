# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--main_ip', default='', type=str)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--num_machines', default=4, type=int)
args = parser.parse_args()

config = {
    'command_file': 'null',
    'commands': 'null',
    'compute_environment': 'LOCAL_MACHINE',
    'deepspeed_config': {},
    'distributed_type': 'MULTI_GPU',
    'downcast_bf16': 'no',
    'dynamo_backend': 'NO',
    'fsdp_config': {},
    'gpu_ids': 'all',
    'machine_rank': args.rank,
    'main_process_ip': args.main_ip,
    'main_process_port': 8080,
    'main_training_function': 'main',
    'megatron_lm_config': {},
    'mixed_precision': 'fp16',
    'num_machines': args.num_machines,
    'num_processes': args.num_machines * 8,
    'rdzv_backend': 'static',
    'same_network': True,
    'tpu_name': 'null',
    'tpu_zone': 'null',
    'use_cpu': False,
}

os.makedirs('/root/.cache/huggingface/accelerate', exist_ok=True)

with open('/root/.cache/huggingface/accelerate/default_config.yaml', 'w') as file:
    yaml.dump(config, file)


