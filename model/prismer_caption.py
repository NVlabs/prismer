# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import torch
import numpy as np

from einops.einops import rearrange
from model.prismer import Prismer


class PrismerCaption(Prismer):
    def forward(self, experts, caption=None, answer=None, train=True, prefix='', inference='generate', k_test=32):
        device = experts['rgb'].device
        if train:
            experts_train = self.expert_encoder(experts)
            experts_train = rearrange(experts_train, 'l b d -> b l d')  # batch_size, num_latents, output_dim

            caption = self.tokenizer(caption, padding='longest', truncation=True, max_length=30, return_tensors="pt").to(device)
            answer_targets = caption.input_ids.masked_fill(caption.input_ids == self.tokenizer.pad_token_id, -100)

            if len(prefix) > 0:
                prompt_length = len(self.tokenizer(prefix).input_ids) - 1  # remove </s> token
                answer_targets[:, :prompt_length] = -100

            answer_output = self.text_decoder(caption.input_ids,
                                              attention_mask=caption.attention_mask,
                                              encoder_hidden_states=experts_train,
                                              labels=answer_targets,
                                              return_dict=True)
            loss = answer_output.loss.mean()
            return loss
        else:
            if inference == 'generate':
                prefixs = [prefix] * experts['rgb'].size(0)
                prefixs = self.tokenizer(prefixs, padding='longest', return_tensors="pt").to(device)
                input_ids = prefixs.input_ids[:, :-1]  # remove </s> token
                attention_masks = prefixs.attention_mask[:, :-1]

                num_beams = 3
                experts_train = self.expert_encoder(experts)
                experts_train = rearrange(experts_train, 'l b d -> b l d')  # batch_size, num_latents, output_dim
                experts_train = experts_train.repeat_interleave(num_beams, dim=0)
                outputs = self.text_decoder.generate(input_ids=input_ids,
                                                     encoder_hidden_states=experts_train,
                                                     attention_mask=attention_masks,
                                                     num_beams=num_beams,
                                                     max_length=20,
                                                     min_length=8)

                captions = []
                for output in outputs:
                    caption = self.tokenizer.decode(output, skip_special_tokens=True)
                    space_idx = 1 if len(prefix) > 0 else 0
                    captions.append(caption[len(prefix) + space_idx:])
                return captions

            elif inference == 'rank':
                device = experts['rgb'].device
                experts_train = self.expert_encoder(experts)
                experts_train = rearrange(experts_train, 'l b d -> b l d')

                answer = [' ' + ans.lower() + '</s>' for ans in answer]
                answer = self.tokenizer(answer, padding='longest', return_tensors='pt', add_special_tokens=False).to(device)

                prefix = [prefix] * experts['rgb'].size(0)
                prefix = self.tokenizer(prefix, padding='longest', return_tensors="pt").to(device)

                start_ids = prefix.input_ids[:, :-1]  # remove </s> token
                attention_masks = prefix.attention_mask[:, :-1]

                start_output = self.text_decoder(start_ids,
                                                 attention_mask=attention_masks,
                                                 encoder_hidden_states=experts_train,
                                                 return_dict=True)

                logits = start_output.logits[:, -1, :]
                answer_first_token = answer.input_ids[:, 0]
                prob_first_token = torch.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
                _, topk_ids = prob_first_token.topk(k_test, dim=1)

                # answer input: [num_caption * k, answer_len]
                answer_input_ids = []
                answer_input_atts = []
                for b, topk_id in enumerate(topk_ids):
                    answer_input_ids.append(answer.input_ids.index_select(dim=0, index=topk_id))
                    answer_input_atts.append(answer.attention_mask.index_select(dim=0, index=topk_id))

                answer_input_ids = torch.cat(answer_input_ids, dim=0)
                answer_input_atts = torch.cat(answer_input_atts, dim=0)

                # repeat encoder's output for top-k answers
                input_ids = torch.cat([tile(start_ids, 0, k_test), answer_input_ids], dim=1).long()
                attention_masks = torch.cat([tile(attention_masks, 0, k_test), answer_input_atts], dim=1)
                experts_train = tile(experts_train, 0, k_test)

                answer_targets = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
                answer_targets[:, :-answer.input_ids.shape[1]] = -100

                output = self.text_decoder(input_ids,
                                           attention_mask=attention_masks,
                                           encoder_hidden_states=experts_train,
                                           labels=answer_targets,
                                           return_dict=True)

                log_probs_sum = -output.loss / torch.sum(answer_targets != -100, dim=-1)
                log_probs_sum = log_probs_sum.view(-1, k_test)

                max_topk_ids = log_probs_sum.argmax(dim=1)
                max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]
                return max_ids


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))

