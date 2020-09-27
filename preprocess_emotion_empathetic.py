# this preprocess file aims at prrocessing data for modeling_JointSentGPT.py

from pytorch_transformers import GPT2Tokenizer
import config.config_emotion_gpt as cfg
from typing import Dict
import os
import re
import json
from tqdm import tqdm
import math
import numpy as np
import torch
from collections import OrderedDict, Counter


### be careful! we starts from idx 1 since we consider pad ar idx 0
old_label_map = {'surprised': 1, 'excited': 2, 'angry': 3, 'proud': 4, 'sad': 5, 'annoyed': 6, 'grateful': 7, 'lonely': 8,
                 'afraid': 9, 'terrified': 10, 'guilty': 11, 'impressed': 12, 'disgusted': 13, 'hopeful': 14, 'confident': 15,
                 'furious': 16, 'anxious': 17, 'anticipating': 18, 'joyful': 19, 'nostalgic': 20, 'disappointed': 21,
                 'prepared': 22, 'jealous': 23, 'content': 24, 'devastated': 25, 'embarrassed': 26, 'caring': 27,
                 'sentimental': 28, 'trusting': 29, 'ashamed': 30, 'apprehensive': 31, 'faithful': 32}

class Input_feature_dialog(object):
    def __init__(self, input_id, input_mask, cls_mask, label_id, fine_grained_emotion_id, coarse_grained_emotion_id,
                 type_id,
                 response_emotion_fine_label, response_emotion_coarse_label, response_emotion_mask):
        self.input_id = input_id
        self.input_mask = input_mask
        self.cls_mask = cls_mask
        self.label_id = label_id
        self.fine_grained_emotion_id = fine_grained_emotion_id
        self.coarse_grained_emotion_id = coarse_grained_emotion_id
        self.response_emotion_fine_label = response_emotion_fine_label
        self.response_emotion_coarse_label = response_emotion_coarse_label
        self.response_emotion_mask = response_emotion_mask
        self.type_id = type_id


def coarse_grained_label_to_id():
    """be careful! we starts from idx 1 since we consider pad ar idx 0
    """
    return {'VAD1': 1, 'VAD2': 2, 'VAD3': 3, 'VAD4': 4, 'VAD5': 5, 'VAD6': 6, 'VAD7': 7}


def _truncate_seq_pair(tokens_hist, tokens_resp, max_length, forward_truncate=False):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_hist) + len(tokens_resp)
        if total_length <= max_length:
            break
        if len(tokens_hist) > len(tokens_resp):
            if forward_truncate:
                tokens_hist = tokens_hist[1:]
            else:
                tokens_hist = tokens_hist[:-1]
        else:
            tokens_resp = tokens_resp[:-1]
    return tokens_hist, tokens_resp


def build_emotion_dialog_data(tokenizer: GPT2Tokenizer, dialog_json_file, output_cache_file, data_type,
                              fine_grained_label_map: Dict[str, int], coarse_grained_label_map: Dict[str, int],
                              max_seq_length=128, cls1_token='[cls1]', cls2_token='[cls2]', speaker0_state=1,
                              speaker1_state=2, speaker_state_pad=0, emotion_pad=0,
                              speaker0_token='<eos0>', speaker1_token='<eos1>', forward_truncate=True):
    features = []

    with open(dialog_json_file, 'r', encoding='utf-8') as f:
        whole_data = json.load(f)
        for data in tqdm(whole_data, desc=f'build emotion {data_type} dialog data'):
            emotion_fine_grained = data['fine_grained_emotion']
            emotion_coarse_grained = data['coarse_grained_emotion']

            # tokenize each utterence
            utterence, token_type_ids = [], []
            for idx, sent in enumerate(data['utterence']):
                sent_token = tokenizer.tokenize(sent) + [speaker0_token if idx % 2 == 0 else speaker1_token]
                token_type_ids.append([speaker0_state if idx % 2 == 0 else speaker1_state] * len(sent_token))
                utterence.append(sent_token)

            # build context and response. Be careful, we only use speaker1 as response
            for i in range(1, len(utterence)):
                if i % 2 == 0:
                    continue

                # get context token and token type id
                context = [token for sent_token in utterence[:i] for token in sent_token]
                type_ids = [type_id for sent_type in token_type_ids[:i] for type_id in sent_type]

                # get context emotion
                context_fine_grained_emotion, context_coarse_grained_emotion = [], []
                for sub_cont_idx in range(i):
                    curr_utter_length = len(utterence[sub_cont_idx])
                    context_fine_grained_emotion.extend(
                        [fine_grained_label_map[emotion_fine_grained[sub_cont_idx]]] * curr_utter_length)
                    context_coarse_grained_emotion.extend(
                        [coarse_grained_label_map[emotion_coarse_grained[sub_cont_idx]]] * curr_utter_length)

                # get response and its emotion, -1 means we consider pad
                response = utterence[i] + [tokenizer.eos_token]
                emotion_labels_coarse = coarse_grained_label_map[emotion_coarse_grained[i]] - 1
                emotion_labels_fine = fine_grained_label_map[emotion_fine_grained[i]] - 1

                # truncate context and response length
                context, response = _truncate_seq_pair(context, response, max_seq_length - 2,
                                                       forward_truncate=forward_truncate)
                truncate_context_len, truncate_resp_len = len(context), len(response)

                # combine context and respsonse according to truncate context and truncate response
                if forward_truncate:
                    type_ids = [speaker_state_pad] * 2 + type_ids[-truncate_context_len:] + [speaker1_state] * (
                        truncate_resp_len)
                    final_fine_grained_emotion = [emotion_pad] * 2 + context_fine_grained_emotion[
                                                                     -truncate_context_len:] + [emotion_pad] * (
                                                             max_seq_length - 2 - truncate_context_len)
                    final_coarse_grained_emotion = [emotion_pad] * 2 + context_coarse_grained_emotion[
                                                                       -truncate_context_len:] + [emotion_pad] * (
                                                               max_seq_length - 2 - truncate_context_len)
                else:
                    type_ids = [speaker_state_pad] * 2 + type_ids[:truncate_context_len] + [speaker1_state] * (
                        truncate_resp_len)
                    final_fine_grained_emotion = [emotion_pad] * 2 + context_fine_grained_emotion[
                                                                     :truncate_context_len] + [emotion_pad] * (
                                                             max_seq_length - 2 - truncate_context_len)
                    final_coarse_grained_emotion = [emotion_pad] * 2 + context_coarse_grained_emotion[
                                                                       :truncate_context_len] + [emotion_pad] * (
                                                               max_seq_length - 2 - truncate_context_len)
                context_id = tokenizer.convert_tokens_to_ids([cls1_token, cls2_token] + context)
                response_id = tokenizer.convert_tokens_to_ids(response)
                input_ids = context_id + response_id

                # pad to max_seq_length
                pad_length = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
                type_ids = type_ids + [speaker_state_pad] * pad_length
                cls1_mask_row = [1, 0] + [1] * (truncate_context_len + truncate_resp_len) + [0] * pad_length
                cls1_mask_col = [1] + [0] * (max_seq_length - 1)
                cls2_mask_row = [0, 1] + [1] * truncate_context_len + [0] * (max_seq_length - 2 - truncate_context_len)
                cls2_mask_col = [0, 1] + [0] * (max_seq_length - 2)
                attention_masks = [1] * (2 + truncate_context_len + truncate_resp_len) + [0] * pad_length

                # pay attention! since before we already -1 to get true label, now we must consider pad at idx 0
                response_emotion_mask = [0] * (truncate_context_len + 2) + [1] * truncate_resp_len + [0] * pad_length
                label_ids = [-1] * (truncate_context_len + 2) + response_id + [-1] * pad_length

                assert len(final_fine_grained_emotion) == max_seq_length
                assert len(final_coarse_grained_emotion) == max_seq_length
                assert len(response_emotion_mask) == max_seq_length
                assert len(input_ids) == max_seq_length
                assert len(attention_masks) == max_seq_length
                assert len(label_ids) == max_seq_length
                assert len(type_ids) == max_seq_length
                assert len(cls1_mask_row) == max_seq_length
                assert len(cls1_mask_col) == max_seq_length
                assert len(cls2_mask_row) == max_seq_length
                assert len(cls2_mask_col) == max_seq_length

                features.append(Input_feature_dialog(input_id=input_ids,
                                                     cls_mask=[cls1_mask_row, cls1_mask_col, cls2_mask_row,
                                                               cls2_mask_col],
                                                     input_mask=attention_masks,
                                                     label_id=label_ids,
                                                     fine_grained_emotion_id=final_fine_grained_emotion,
                                                     coarse_grained_emotion_id=final_coarse_grained_emotion,
                                                     response_emotion_coarse_label=emotion_labels_coarse,
                                                     response_emotion_fine_label=emotion_labels_fine,
                                                     response_emotion_mask=response_emotion_mask,
                                                     type_id=type_ids))

    torch.save(features, output_cache_file)


def cache_test_examples(tokenizer: GPT2Tokenizer, test_json_file: str, output_file: str, max_seq_length,
                        fine_grained_label_map: Dict[str, int], coarse_grained_label_map: Dict[str, int],
                        speaker0_token='<eos0>', speaker1_token='<eos1>', emotion_pad=0,
                        cls1_token='[cls1]', cls2_token='[cls2]', forward_truncate=True,
                        speaker0_state=1, speaker1_state=2, speaker_state_pad=0):
    json_test = []

    with open(test_json_file, 'r', encoding='utf-8') as f:
        whole_data = json.load(f)
        for data in tqdm(whole_data, desc='extract test id'):
            emotion_fine_grained = data['fine_grained_emotion']
            emotion_coarse_grained = data['coarse_grained_emotion']
            label = data['label']

            # tokenize each utterence
            utterence, token_type_ids = [], []
            for idx, sent in enumerate(data['utterence']):
                sent_token = tokenizer.tokenize(sent) + [speaker0_token if idx % 2 == 0 else speaker1_token]
                token_type_ids.append([speaker0_state if idx % 2 == 0 else speaker1_state] * len(sent_token))
                utterence.append(sent_token)

            # get context topk emotion
            for i in range(1, len(utterence)):
                if i % 2 == 0:
                    continue

                new_data = {}
                context_token_flatten = [token for sent_token in utterence[:i] for token in sent_token]
                type_ids = [type_id for sent_type in token_type_ids[:i] for type_id in sent_type]

                ref_token = utterence[i]

                # get context topk emotion
                context_fine_grained_emotion, context_coarse_grained_emotion = [], []
                for sub_cont_idx in range(i):
                    curr_utter_length = len(utterence[sub_cont_idx])
                    context_fine_grained_emotion.extend(
                        [fine_grained_label_map[emotion_fine_grained[sub_cont_idx]]] * curr_utter_length)
                    context_coarse_grained_emotion.extend(
                        [coarse_grained_label_map[emotion_coarse_grained[sub_cont_idx]]] * curr_utter_length)

                context_token_flatten, ref_token = _truncate_seq_pair(context_token_flatten, ref_token,
                                                                      max_seq_length - 2,
                                                                      forward_truncate=forward_truncate)
                truncate_context_len = len(context_token_flatten)

                if forward_truncate:
                    context_fine_grained_emotion = [emotion_pad] * 2 + context_fine_grained_emotion[
                                                                       -truncate_context_len:]
                    context_coarse_grained_emotion = [emotion_pad] * 2 + context_coarse_grained_emotion[
                                                                         -truncate_context_len:]
                    type_ids = [speaker_state_pad] * 2 + type_ids[-truncate_context_len:]
                else:
                    context_fine_grained_emotion = [emotion_pad] * 2 + context_fine_grained_emotion[
                                                                       :truncate_context_len]
                    context_coarse_grained_emotion = [emotion_pad] * 2 + context_coarse_grained_emotion[
                                                                         :truncate_context_len]
                    type_ids = [speaker_state_pad] * 2 + type_ids[:truncate_context_len]
                input_ids = tokenizer.convert_tokens_to_ids([cls1_token, cls2_token] + context_token_flatten)

                cls1_mask_row = [1, 0] + [1] * truncate_context_len
                cls1_mask_col = [1, 0] + [0] * truncate_context_len
                cls2_mask_row = [0, 1] + [1] * truncate_context_len
                cls2_mask_col = [0, 1] + [0] * truncate_context_len

                assert len(input_ids) == len(context_coarse_grained_emotion)
                assert len(type_ids) == len(context_coarse_grained_emotion)
                assert len(cls1_mask_col) == len(context_coarse_grained_emotion)
                assert len(cls1_mask_row) == len(context_coarse_grained_emotion)

                new_data.update({'input_ids': input_ids,
                                 'token_type_ids': type_ids,
                                 'fine_grained_emotion_ids': context_fine_grained_emotion,
                                 'coarse_grained_emotion_ids': context_coarse_grained_emotion,
                                 'cls_mask': [cls1_mask_row, cls1_mask_col, cls2_mask_row, cls2_mask_col],
                                 'label': label})

                json_test.append(new_data)

    print("test num: ", len(json_test))
    f_w = open(output_file, 'w', encoding='utf-8')
    json.dump(json_test, f_w)


if __name__ == "__main__":
    tokenizer_gpt = GPT2Tokenizer.from_pretrained(cfg.model_path)
    tokenizer_gpt.add_special_tokens(OrderedDict(cfg.SPECIAL_tokens))

    fine_grained_label_map = old_label_map
    coarse_grained_label_map = coarse_grained_label_to_id()

    for d_type in ['train', 'valid', 'test']:
        dialog_json = os.path.join(cfg.data_dir, f"parsed_emotion_32_VAD7_{d_type}_new.json")
        output_cache_file = os.path.join(cfg.cache_dir, f"cache_JointSentGPT_Joint_latent_32_VAD7_{d_type}_len{cfg.max_sequence_length}")
        build_emotion_dialog_data(tokenizer=tokenizer_gpt,
                                  dialog_json_file=dialog_json,
                                  output_cache_file=output_cache_file,
                                  data_type=d_type,
                                  fine_grained_label_map=fine_grained_label_map,
                                  coarse_grained_label_map=coarse_grained_label_map,
                                  max_seq_length=cfg.max_sequence_length)

        output_test_decoding_file = os.path.join(cfg.cache_dir, f"cache_JointSentGPT_decode_Joint_latent_32_VAD7_{d_type}_len{cfg.max_sequence_length}.json")
        cache_test_examples(tokenizer=tokenizer_gpt,
                            test_json_file=dialog_json,
                            output_file=output_test_decoding_file,
                            max_seq_length=cfg.max_sequence_length,
                            fine_grained_label_map=fine_grained_label_map,
                            coarse_grained_label_map=coarse_grained_label_map)

