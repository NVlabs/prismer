# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import numpy as np
import torch
import torch.nn.functional as F

from einops.einops import rearrange
from model.prismer import Prismer


class PrismerVQA(Prismer):
    def forward(self, experts, question, answer=None, weights=None, train=True, inference='rank', k_test=128):
        device = experts['rgb'].device
        question = ['<s>' + ques.capitalize() for ques in question]
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35,
                                  add_special_tokens=False, return_tensors="pt").to(device)
        
        if train:
            experts_train = self.expert_encoder(experts)
            experts_train = rearrange(experts_train, 'l b d -> b l d')  # batch_size, num_latents, output_dim

            answer = [' ' + ans.capitalize() + '</s>' for ans in answer]
            answer = self.tokenizer(answer, padding='longest', return_tensors="pt", add_special_tokens=False).to(device)

            input_ids = torch.cat([question.input_ids, answer.input_ids], dim=1).long()
            attention_mask = torch.cat([question.attention_mask, answer.attention_mask], dim=1)

            answer_targets = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
            answer_targets[:, :-answer.input_ids.shape[1]] = -100

            answer_output = self.text_decoder(input_ids,
                                              attention_mask=attention_mask,
                                              encoder_hidden_states=experts_train,
                                              labels=answer_targets,
                                              return_dict=True)
            loss = weights * answer_output.loss
            loss = loss.mean()
            return loss
        else:
            if inference == 'generate':
                num_beams = 3
                input_ids = question.input_ids
                attention_masks = question.attention_mask

                experts_train = self.expert_encoder(experts)
                experts_train = rearrange(experts_train, 'l b d -> b l d')  # batch_size, num_latents, output_dim
                experts_train = experts_train.repeat_interleave(num_beams, dim=0)
                outputs = self.text_decoder.generate(input_ids=input_ids,
                                                     encoder_hidden_states=experts_train,
                                                     attention_mask=attention_masks,
                                                     max_length=input_ids.shape[1] + 10,
                                                     min_length=input_ids.shape[1] + 2,
                                                     num_beams=num_beams,
                                                     length_penalty=-1)
                answers = []
                for i in range(len(outputs)):
                    answer = self.tokenizer.decode(outputs[i, len(input_ids[i]):], skip_special_tokens=True)
                    answers.append(answer.lower().strip())
                return answers

            elif inference == 'rank':
                experts_train = self.expert_encoder(experts)
                experts_train = rearrange(experts_train, 'l b d -> b l d')

                answer = [' ' + ans.capitalize() + '</s>' for ans in answer]
                answer = self.tokenizer(answer, padding='longest', return_tensors='pt', add_special_tokens=False).to(device)

                start_ids = question.input_ids
                attention_masks = question.attention_mask

                start_output = self.text_decoder(start_ids,
                                                 attention_mask=attention_masks,
                                                 encoder_hidden_states=experts_train,
                                                 return_dict=True)

                logits = start_output.logits[:, -1, :]
                answer_first_token = answer.input_ids[:, 0]
                prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
                _, topk_ids = prob_first_token.topk(k_test, dim=1)

                # answer input: [num_question * k, answer_len]
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
