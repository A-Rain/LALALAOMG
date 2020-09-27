# this training and eval file corresponding to preprocess_emotion_empathetic_v11.py, modeling_JointSentiGPT_v2.py and config_emotion_gpt_v9.py

import os
import re
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from typing import List, Dict
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import SequentialSampler, RandomSampler, DataLoader
import argparse

from metric import compute_metrics
from Effective_Parallel import DataParallelCriterion, DataParallelModel
import config.config_emotion_gpt as cfg_gpt
from preprocess_emotion_empathetic import Input_feature_dialog
from JointSentiGPT.modeling_JointSentiGPT import JointSentiGPT2Model
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import GPT2Config, GPT2Tokenizer


def set_seed(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)


def load_cache_examples(cfg, data_type):
    print(f"***** load {data_type} data cache *****")
    file_path = os.path.join(cfg.cache_dir, f"cache_JointSentGPT_Joint_latent_32_VAD7_{data_type}_len{cfg.max_sequence_length}")

    features = torch.load(file_path)

    input_ids = torch.Tensor([f.input_id for f in features]).long()
    token_type_ids = torch.Tensor([f.type_id for f in features]).long()
    input_masks = torch.Tensor([f.input_mask for f in features]).long()
    cls_masks = torch.Tensor([f.cls_mask for f in features]).long()  # [Bsz, 2, seq_length]
    labels = torch.Tensor([f.label_id for f in features]).long()
    response_emotion_masks = torch.Tensor([f.response_emotion_mask for f in features]).long()

    if cfg.emotion_cls == 'coarse':
        emotion_ids = torch.Tensor([f.coarse_grained_emotion_id for f in features]).long()
        response_emotion_label = torch.Tensor([f.response_emotion_coarse_label for f in features]).long()
    else:
        emotion_ids = torch.Tensor([f.fine_grained_emotion_id for f in features]).long()
        response_emotion_label = torch.Tensor([f.response_emotion_fine_label for f in features]).long()

    dataset = TensorDataset(input_ids, token_type_ids, input_masks, cls_masks, labels, emotion_ids,
                            response_emotion_masks, response_emotion_label)
    if data_type == 'train':
        Bsz, sampler = cfg.train_batch_size, RandomSampler(dataset)
    else:
        Bsz, sampler = cfg.eval_batch_size, SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=Bsz, drop_last=True)
    return dataloader


def calc_loss(cfg, step, lm_logits, emotion_logits, src_hidden, target_hidden, lm_labels: torch.Tensor,
              emotion_labels: torch.Tensor, loss_cls_fct, loss_dist_fct, threshold=1e-8):
    """
    src_hidden: [Bsz, hidden_size], target_hidden: [Bsz, hidden_size]
    """

    shift_lm_labels = lm_labels[..., 1:].contiguous()
    loss_nll = loss_cls_fct(lm_logits, shift_lm_labels.view(-1))

    loss_hid_dist, loss_emotion = 0, 0
    if step < cfg.only_nll_step:
        pass
    else:
        loss_emotion = loss_cls_fct(emotion_logits, emotion_labels.view(-1))
        if step >= cfg.calc_hid_dist_step:
            loss_hid_dist = loss_dist_fct(src_hidden, target_hidden)

        if cfg.use_hinge:
            loss_hid_dist = max(0, loss_hid_dist - cfg.hinge_phi)

    loss = loss_nll + cfg.alpha * loss_emotion + cfg.beta * loss_hid_dist

    return loss, loss_nll, loss_emotion, loss_hid_dist


def train(cfg, model: JointSentiGPT2Model, train_dataloader: DataLoader, val_dataloader: DataLoader):
    steps_per_batch = len(train_dataloader)
    t_total = steps_per_batch * cfg.num_train_epochs
    warmup_steps = int(cfg.warmup_proportion * t_total)

    cfg.calc_hid_dist_step = cfg.calc_hid_dist_step * steps_per_batch
    cfg.only_nll_step = cfg.only_nll_step * steps_per_batch
    cfg.whole_step = t_total

    print("***** Running training *****")
    print(f"temperature = {cfg.temperature}")
    print(f"alpha = {cfg.alpha}")
    print(f"beta = {cfg.beta}")
    print(f"leak_emotion_step = {cfg.leak_emotion_step}")
    print(f"temperature_update = {cfg.adapt}")
    print(f"parallel = {cfg.parallel}")
    print(f"seed = {cfg.seed}")
    print(f"max_seq_length = {cfg.max_sequence_length}")
    print(f"emotion type = {cfg.emotion_cls}")
    print(f"num epochs = {cfg.num_train_epochs}")
    print(f"whole training steps = {t_total}")
    print(f"warmup steps = {warmup_steps}")
    print(f"train batch size = {cfg.train_batch_size}")
    print(f"learning rate = {cfg.learning_rate}")

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': cfg.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    record_txt = open(os.path.join(cfg.save_dir, "record.log"), 'w', encoding='utf-8')
    distribution_distance_record = open(os.path.join(cfg.save_dir, "dist_record.log"), 'w', encoding='utf-8')

    if cfg.parallel:
        loss_cls_fct = DataParallelCriterion(torch.nn.CrossEntropyLoss(ignore_index=-1))
        if cfg.dist_loss == 'mse':
            loss_dist_fct = DataParallelCriterion(torch.nn.MSELoss())
        elif cfg.dist_loss == 'cos':
            loss_dist_fct = DataParallelCriterion(torch.nn.CosineEmbeddingLoss())

    else:
        loss_cls_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        if cfg.dist_loss == 'mse':
            loss_dist_fct = torch.nn.MSELoss()
        elif cfg.dist_loss == 'cos':
            loss_dist_fct = torch.nn.CosineEmbeddingLoss()

    model.zero_grad()
    global_step = 0
    for epo in range(1, cfg.num_train_epochs + 1):
        model.train()
        tqdm_bar = tqdm(train_dataloader, desc="Training")
        avg_train_loss, avg_train_loss_nll, avg_train_loss_emotion, avg_train_loss_dist, step = 0, 0, 0, 0, 0
        step_calc_dist, step_calc_emotion = 0, 0
        for batch in tqdm_bar:
            batch = tuple(t.to(cfg.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'token_type_ids': batch[1],
                      'attention_mask': batch[2],
                      'cls_mask': batch[3],
                      'emotion_ids': batch[5],
                      'gold_response_emotion_masks': batch[6],
                      'decoding': False,
                      'step': global_step}

            outputs = model(**inputs)
            if cfg.parallel:
                # make the right input for parallel criterion
                lm_logits, emotion_logits, src_hidden_states, target_hidden_states = list(zip(*outputs))[:4]
                lm_logits = [(element,) for element in lm_logits]
                emotion_logits = [(element,) for element in emotion_logits]
                if cfg.normalize:
                    src_hidden_states = [(F.normalize(element, p=2, dim=-1),) for element in src_hidden_states]
                    target_hidden_states = F.normalize(torch.cat(target_hidden_states, dim=-1), p=2, dim=-1).detach()
                else:
                    src_hidden_states = [(element,) for element in src_hidden_states]
                    target_hidden_states = torch.cat(target_hidden_states, dim=-1).detach()

            else:
                lm_logits, emotion_logits, src_hidden_states, target_hidden_states = outputs[:4]
                if cfg.normalize:
                    src_hidden_states = F.normalize(src_hidden_states, p=2, dim=-1)
                    target_hidden_states = F.normalize(target_hidden_states, p=2, dim=-1).detach()

            loss, loss_nll, loss_emotion, loss_hid_dist = calc_loss(cfg, global_step, lm_logits, emotion_logits,
                                                                    src_hidden_states, target_hidden_states, batch[4],
                                                                    batch[7], loss_cls_fct, loss_dist_fct)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            if global_step >= cfg.only_nll_step:
                step_calc_emotion += 1
            if global_step >= cfg.calc_hid_dist_step:
                distribution_distance_record.writelines(str(loss_hid_dist.item()) + '\n')
                step_calc_dist += 1

            avg_train_loss += loss.item()
            avg_train_loss_nll += loss_nll.item()
            avg_train_loss_emotion += loss_emotion.item() if isinstance(loss_emotion, torch.Tensor) else 0
            avg_train_loss_dist += loss_hid_dist.item() if isinstance(loss_hid_dist, torch.Tensor) else 0
            step += 1
            global_step += 1

            loss_dist = 0 if step_calc_dist == 0 else avg_train_loss_dist / step_calc_dist
            loss_emotion = 0 if step_calc_emotion == 0 else avg_train_loss_emotion / step_calc_emotion
            tqdm_bar.desc = f"epoch:[{epo}/{cfg.num_train_epochs}],step:[{step}/{steps_per_batch}],loss:{avg_train_loss / step:.4f},nll:{avg_train_loss_nll / step:.4f}," \
                f"emotion:{loss_emotion:.4f},dist:{loss_dist:.4f}"

        if global_step >= cfg.calc_hid_dist_step:
            distribution_distance_record.writelines("\n")

        avg_train_loss = avg_train_loss / step
        avg_train_loss_nll = avg_train_loss_nll / step
        avg_train_loss_emotion = 0 if step_calc_emotion == 0 else avg_train_loss_emotion / step_calc_emotion
        avg_train_loss_dist = 0 if step_calc_dist == 0 else avg_train_loss_dist / step_calc_dist

        dev_loss_nll, dev_loss_emotion_hist_resp, dev_loss_hid_dist = validation(cfg, model, val_dataloader)
        print(
            f"in epoch {epo}, dev_loss_nll: {dev_loss_nll:.4f}, dev_loss_emotion: {dev_loss_emotion_hist_resp:.4f}, dev_loss_hid_dist: {dev_loss_hid_dist:.4f}, train_loss: {avg_train_loss:.4f}, train_loss_dist: {avg_train_loss_dist:.4f}")

        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                   os.path.join(cfg.save_dir, f"epo{epo}.pt"))
        record_txt.writelines(f"epoch: {epo}\t\t"
                              f"train_loss: {avg_train_loss:.4f}\t\ttrain_hid_dist: {avg_train_loss_dist:.4f}\t\ttrain_emotion: {avg_train_loss_emotion:.4f}\t\ttrain_nll: {avg_train_loss_nll:.4f}\t\t"
                              f"dev_emotion: {dev_loss_emotion_hist_resp:.4f}\t\tdev_hid_dist: {dev_loss_hid_dist:.4f}\t\tdev loss nll: {dev_loss_nll}\n")


def validation(cfg, model: JointSentiGPT2Model, val_dataloader: DataLoader, decoding=False):
    dev_loss_nll, dev_loss_emotion, dev_loss_hid_dist, count = 0, 0, 0, 0
    mode = cfg.parallel and (not decoding)

    loss_cls_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
    if cfg.dist_loss == 'mse':
        loss_dist_fct = torch.nn.MSELoss()
    elif cfg.dist_loss == 'cos':
        loss_dist_fct = torch.nn.CosineEmbeddingLoss()

    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_dataloader):
            batch = tuple(t.to(cfg.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'token_type_ids': batch[1],
                      'attention_mask': batch[2],
                      'cls_mask': batch[3],
                      'emotion_ids': batch[5],
                      'gold_response_emotion_masks': batch[6],
                      'decoding': decoding}

            if mode:
                outputs = model.module.validate(**inputs)
            else:
                outputs = model.validate(**inputs)

            lm_logits, emotion_logits, src_hidden_states, target_hidden_states = outputs

            if cfg.normalize:
                src_hidden_states = F.normalize(src_hidden_states, p=2, dim=-1)
                target_hidden_states = F.normalize(target_hidden_states, dim=-1)

            loss, loss_nll, loss_emotion, loss_hid_dist = calc_loss(cfg, np.Inf, lm_logits, emotion_logits,
                                                                    src_hidden_states, target_hidden_states.detach(), batch[4],
                                                                    batch[7], loss_cls_fct, loss_dist_fct)

            dev_loss_nll += loss_nll.item()
            dev_loss_emotion += loss_emotion.item()
            dev_loss_hid_dist += loss_hid_dist.item() if isinstance(loss_hid_dist, torch.Tensor) else 0
            count += 1

    dev_loss_nll = dev_loss_nll / count
    dev_loss_emotion = dev_loss_emotion / count
    dev_loss_hid_dist = dev_loss_hid_dist / count

    return dev_loss_nll, dev_loss_emotion, dev_loss_hid_dist


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(cfg, model: JointSentiGPT2Model, tokenizer: GPT2Tokenizer, context_token: torch.Tensor,
                    token_type: torch.Tensor, context_emotion: torch.Tensor, cls_mask: torch.Tensor,
                    emotion_pad=0, speaker1_state=2, decoding_strategy='sampling'):
    cls_mask_extra = torch.LongTensor([[[1], [0], [0], [0]]]).to(cfg.device)

    context_len = context_token.shape[1]
    generated = context_token

    past, pred_response_emotion = None, None
    result = []
    for step in range(cfg.max_decode_length):
        inputs = {'input_ids': generated,
                  'token_type_ids': token_type,
                  'emotion_ids': context_emotion,
                  'pred_response_emotion_vector': pred_response_emotion,
                  'cls_mask': cls_mask,
                  'past': past,
                  'decoding': True}
        outputs = model.decoding(
            **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
        pred_response_emotion, past = outputs[1:]
        next_token_logits = outputs[0][0, -1, :] / cfg.sampling_temperature
        if decoding_strategy == 'sampling':
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=cfg.top_k, top_p=cfg.top_p)
            prob = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(prob, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1)
            next_token = next_token.unsqueeze(0)

        if next_token.item() == tokenizer.eos_token_id and step >= cfg_gpt.min_decode_length:
            break

        result.append(next_token.item())
        generated = next_token.unsqueeze(0)
        token_type = torch.LongTensor([[speaker1_state]]).to(cfg.device)
        cls_mask = torch.cat((cls_mask, cls_mask_extra), dim=-1)

    # generated = generated[0, context_len:].tolist()
    result = [token_id for token_id in result if token_id not in cfg.special_id_list]
    text = tokenizer.decode(result, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    text = text.replace("\n", "").replace("\r", "")
    return text


def evaluation(cfg, model: JointSentiGPT2Model, tokenizer: GPT2Tokenizer, test_dataset: List[Dict],
               test_dataloader: DataLoader, d_type):
    src, hypothesis, dialog_situation_label = [], [], []
    model.eval()
    with torch.no_grad():
        for one_batch in tqdm(test_dataset, desc=f'decoding {d_type}...'):
            context_id = one_batch['input_ids']
            if d_type == 'test':
                dialog_situation_label.append(one_batch['label'])
            src.append(tokenizer.decode(context_id))

            # [seq_length] -> [1,seq_length]
            token_type_id = torch.LongTensor(one_batch['token_type_ids']).unsqueeze(0).to(cfg.device)
            if cfg.emotion_cls == 'coarse':
                emotion_id = torch.LongTensor(one_batch['coarse_grained_emotion_ids']).unsqueeze(0).to(cfg.device)
            else:
                emotion_id = torch.LongTensor(one_batch['fine_grained_emotion_ids']).unsqueeze(0).to(cfg.device)
            context_id = torch.LongTensor(context_id).unsqueeze(0).to(cfg.device)

            # [2, seq_length] -> [1, 2, seq_length]
            cls_mask = torch.LongTensor(one_batch['cls_mask']).unsqueeze(0).to(cfg.device)

            hyp = sample_sequence(cfg, model, tokenizer, context_id, token_type_id, emotion_id, cls_mask,
                                  decoding_strategy=cfg.decoding_method)
            hypothesis.append(hyp)

    suffix = 'greedy' if cfg.decoding_method == 'greedy' else f'sampling_topk{cfg.top_k}_topp{cfg.top_p}_tau{cfg.sampling_temperature}'
    if d_type in ['train', 'valid']:
        hyp_file = os.path.join(cfg.save_dir,
                                f"epo{cfg.best_epoch}_{d_type}_hyp{cfg.min_decode_length}_{cfg.max_decode_length}_{suffix}.txt")
        with open(hyp_file, 'w', encoding='utf-8') as f_hyp:
            for hypo in hypothesis:
                f_hyp.writelines(hypo.strip() + '\n')
    else:
        hyp_file = os.path.join(cfg.save_dir,
                                f"epo{cfg.best_epoch}_test_hyp{cfg.min_decode_length}_{cfg.max_decode_length}_{suffix}.txt")
        check_file = os.path.join(cfg.save_dir,
                                  f"epo{cfg.best_epoch}_test_check{cfg.min_decode_length}_{cfg.max_decode_length}_{suffix}.txt")

        if cfg.lower:
            result_file = os.path.join(cfg.save_dir,
                                       f"epo{cfg.best_epoch}_result{cfg.min_decode_length}_{cfg.max_decode_length}_lower_{suffix}.json")
        else:
            result_file = os.path.join(cfg.save_dir,
                                       f"epo{cfg.best_epoch}_result{cfg.min_decode_length}_{cfg.max_decode_length}_{suffix}.json")

        ref_txt = cfg.ref_file

        with open(check_file, 'w', encoding='utf-8') as f_check, open(hyp_file, 'w', encoding='utf-8') as f_hyp, \
                open(ref_txt, 'r', encoding='utf-8') as f_r:
            for dialog_history, situation_label, hypo, ref in zip(src, dialog_situation_label, hypothesis, f_r):
                f_check.writelines(f"situation: {situation_label}\n")
                f_check.writelines(f"history: {dialog_history}\n")
                f_check.writelines(f"hyp: {hypo.strip()}\n")
                f_check.writelines(f"ref: {ref.strip()}\n")
                f_check.writelines("\n")

                f_hyp.writelines(hypo.strip() + '\n')

        # compute regular metric
        result = compute_metrics(hyp_file=hyp_file, ref_file=ref_txt, glove_path=cfg.glove_path, lower=cfg.lower,
                                 space_token=False)
        # compute perplexity
        NLL_Loss, _, _ = validation(cfg, model, test_dataloader, decoding=True)
        ppl = np.exp(NLL_Loss)
        print(f"perplexity: {ppl:.6f}")
        result.update({"perplexity": ppl})

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f)


def get_best_epoch(record_log):
    best_loss, best_epoch = 9999, -1
    with open(record_log, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip().split("\t\t")
            dev_loss = float(line[-1].replace("dev loss nll: ", ""))
            if dev_loss < best_loss:
                best_loss, best_epoch = dev_loss, idx + 1
    return best_epoch


def get_special_token_ids(cfg, tokenizer: GPT2Tokenizer):
    special_id_list = []
    for key, value in cfg.SPECIAL_tokens.items():
        if key == 'additional_special_tokens':
            special_id_list.extend(value)
        else:
            special_id_list.append(value)
    return tokenizer.convert_tokens_to_ids(special_id_list)


def main():
    set_seed(cfg_gpt)

    device = torch.device("cuda" if cfg_gpt.use_gpu and torch.cuda.is_available() else 'cpu')
    cfg_gpt.device = device
    local_dir = f"JointSentGPT_32_VAD7_seed{cfg_gpt.seed}_lr{cfg_gpt.learning_rate}_len{cfg_gpt.max_sequence_length}_batch{cfg_gpt.train_batch_size}_epoch{cfg_gpt.num_train_epochs}" \
        f"_warmup{cfg_gpt.warmup_proportion}_{cfg_gpt.emotion_cls}_alpha{cfg_gpt.alpha}_beta{cfg_gpt.beta}_hidLoss{cfg_gpt.dist_loss}_hinge{cfg_gpt.hinge_phi if cfg_gpt.use_hinge else 0}"
    cfg_gpt.save_dir = os.path.join(cfg_gpt.save_dir, local_dir)
    if not os.path.exists(cfg_gpt.save_dir):
        os.mkdir(cfg_gpt.save_dir)
    print("save_dir: ", cfg_gpt.save_dir)

    # prepare path for evaluation
    cfg_gpt.ref_file = os.path.join(cfg_gpt.data_dir, f'ref.txt')

    # prepare config
    config = GPT2Config.from_pretrained(cfg_gpt.model_path)
    emo_num = 7 if cfg_gpt.emotion_cls == 'coarse' else 32
    setattr(config, 'emotion_size', emo_num + 1)
    setattr(config, 'alpha', cfg_gpt.alpha)
    setattr(config, 'temperature', cfg_gpt.temperature)
    setattr(config, 'leak_emotion_step', cfg_gpt.leak_emotion_step)
    setattr(config, 'emotion_label_num', emo_num)
    print(config)

    # build tokenizer
    print("load tokenizer ...")
    tokenizer = GPT2Tokenizer.from_pretrained(cfg_gpt.model_path)
    tokenizer.add_special_tokens(OrderedDict(cfg_gpt.SPECIAL_tokens))
    cfg_gpt.special_id_list = get_special_token_ids(cfg_gpt, tokenizer)

    # build model
    print("load model ...")
    model = JointSentiGPT2Model.from_pretrained(cfg_gpt.model_path, config=config)
    # reshape vocab size
    new_vocab_size = len(tokenizer)
    model.resize_token_embeddings(new_vocab_size)
    model.to(cfg_gpt.device)

    if cfg_gpt.do_train:
        train_loader = load_cache_examples(cfg_gpt, 'train')
        dev_loader = load_cache_examples(cfg_gpt, 'valid')
        if cfg_gpt.parallel:
            print("use parallel")
            model_parallel = DataParallelModel(model)
            train(cfg_gpt, model_parallel, train_loader, dev_loader)
        else:
            train(cfg_gpt, model, train_loader, dev_loader)

    if cfg_gpt.do_eval:
        print("begin decoding ...")
        # load dialog context as test input
        file_test = open(os.path.join(cfg_gpt.cache_dir,
                                      f"cache_JointSentGPT_decode_Joint_latent_32_VAD7_test_len{cfg_gpt.max_sequence_length}.json"),
                         'r', encoding='utf-8')
        test_intput_for_generate = json.load(file_test)

        # # load dialog context as train input
        # file_train = open(os.path.join(cfg_gpt.cache_dir, f"cache_decode_Joint_32_VAD7_train_len128_seed{cfg_gpt.seed}.json"), 'r', encoding='utf-8')
        # train_intput_for_generate = json.load(file_train)

        # # load dialog context as valid input
        # file_valid = open(os.path.join(cfg_gpt.cache_dir, f"cache_decode_Joint_32_VAD7_valid_len128_seed{cfg_gpt.seed}.json"), 'r', encoding='utf-8')
        # valid_intput_for_generate = json.load(file_valid)

        best_epoch = get_best_epoch(os.path.join(cfg_gpt.save_dir, "record.log"))
        cfg_gpt.best_epoch = best_epoch
        model.load_state_dict(torch.load(os.path.join(cfg_gpt.save_dir, f"epo{best_epoch}.pt")))

        # load cache test loader used for calculate perplexity
        test_dataloader = load_cache_examples(cfg_gpt, 'test')

        evaluation(cfg_gpt, model, tokenizer, test_intput_for_generate, test_dataloader, 'test')
        # evaluation(cfg_gpt, model, tokenizer, train_intput_for_generate, test_dataloader, 'train')
        # evaluation(cfg_gpt, model, tokenizer, valid_intput_for_generate, test_dataloader, 'valid')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1024, type=int)
    parser.add_argument("--emotion-cls", default='coarse', type=str)
    parser.add_argument("--device-ids", default='2,3', type=str)
    parser.add_argument("--lr", default=7e-5, type=float)
    parser.add_argument("--epoch", default=8, type=int)
    parser.add_argument("--lower", default=True, type=bool)
    parser.add_argument("--do-train", action='store_true')
    parser.add_argument("--do-eval", action='store_true')
    parser.add_argument("--decoding", default='sampling', type=str)
    parser.add_argument("--parallel", action='store_true')
    args = parser.parse_args()

    cfg_gpt.seed = args.seed
    cfg_gpt.emotion_cls = args.emotion_cls
    cfg_gpt.gpu_id = args.device_ids
    cfg_gpt.lower = args.lower
    cfg_gpt.num_train_epochs = args.epoch
    cfg_gpt.learning_rate = args.lr
    cfg_gpt.do_train = args.do_train
    cfg_gpt.do_eval = args.do_eval
    cfg_gpt.decoding_method = args.decoding
    cfg_gpt.parallel = args.parallel

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg_gpt.gpu_id)
    main()
