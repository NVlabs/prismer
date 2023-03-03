# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import json
import torch.nn as nn

from model.modules.vit import load_encoder
from model.modules.roberta import load_decoder
from transformers import RobertaTokenizer, RobertaConfig


class Prismer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = {'rgb': 3}
        for exp in config['experts']:
            if exp in ['depth', 'edge']:
                self.experts[exp] = 1
            elif exp in ['normal']:
                self.experts[exp] = 3
            elif 'seg' in exp:
                self.experts['seg'] = 64
            elif exp in ['obj_detection', 'ocr_detection']:
                self.experts[exp] = 64

        prismer_config = json.load(open('configs/prismer.json', 'r'))[config['prismer_model']]
        roberta_config = RobertaConfig.from_dict(prismer_config['roberta_model'])

        self.tokenizer = RobertaTokenizer.from_pretrained(prismer_config['roberta_model']['model_name'])
        self.expert_encoder = load_encoder(prismer_config['vit_model'], experts=self.experts, image_resolution=config['image_resolution'])
        self.text_decoder = load_decoder(prismer_config['roberta_model']['model_name'], config=roberta_config)

        self.prepare_to_train(config['freeze'])
        self.ignored_modules = self.get_ignored_modules(config['freeze'])
    
    def prepare_to_train(self, mode='none'):
        for name, params in self.named_parameters():
            if mode == 'freeze_lang':
                if 'encoder.layer' in name and all(key not in name for key in ['1.self', '1.output', 'adaptor']):
                    params.requires_grad = False
                else:
                    params.requires_grad = True
            elif mode == 'freeze_vision':
                if 'transformer.resblocks' in name and 'adaptor' not in name:
                    params.requires_grad = False
                else:
                    params.requires_grad = True
            elif mode == 'freeze_lang_vision':
                if 'encoder.layer' in name and all(key not in name for key in ['1.self', '1.output', 'adaptor']):
                    params.requires_grad = False
                elif 'transformer.resblocks' in name and 'adaptor' not in name:
                    params.requires_grad = False
                else:
                    params.requires_grad = True
            else:
                params.requires_grad = True

    def get_ignored_modules(self, mode='none'):
        ignored_modules = []
        if mode == 'freeze_lang':
            for l in range(len(self.text_decoder.roberta.encoder.layer)):
                ignored_modules += [
                    self.text_decoder.roberta.encoder.layer[l][0].attention,
                    self.text_decoder.roberta.encoder.layer[l][0].intermediate,
                    self.text_decoder.roberta.encoder.layer[l][0].output,
                ]
        elif mode == 'freeze_vision':
            for l in range(len(self.expert_encoder.transformer.resblocks)):
                ignored_modules += [
                    self.expert_encoder.transformer.resblocks[l][0].attn,
                    self.expert_encoder.transformer.resblocks[l][0].mlp,
                    self.expert_encoder.transformer.resblocks[l][0].ln_1,
                    self.expert_encoder.transformer.resblocks[l][0].ln_2,
                ]
        elif mode == 'freeze_lang_vision':
            for l in range(len(self.text_decoder.roberta.encoder.layer)):
                ignored_modules += [
                    self.text_decoder.roberta.encoder.layer[l][0].attention,
                    self.text_decoder.roberta.encoder.layer[l][0].intermediate,
                    self.text_decoder.roberta.encoder.layer[l][0].output,
                ]
            for l in range(len(self.expert_encoder.transformer.resblocks)):
                ignored_modules += [
                    self.expert_encoder.transformer.resblocks[l][0].attn,
                    self.expert_encoder.transformer.resblocks[l][0].mlp,
                    self.expert_encoder.transformer.resblocks[l][0].ln_1,
                    self.expert_encoder.transformer.resblocks[l][0].ln_2,
                ]
        else:
            ignored_modules = None
        return ignored_modules

