import os
import numpy as np
import itertools
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import corpus_bleu

from typing import List


class Embedding(object):
    def __init__(self, glove_path):
        self.m = KeyedVectors.load(os.path.join(glove_path, 'glove.6B.300d.model.bin'), mmap='r')
        self.unk = self.m.vectors.mean(axis=0)

    @property
    def w2v(self):
        return np.concatenate((self.m.syn0, self.unk[None, :]), axis=0)

    def __getitem__(self, key):
        try:
            return self.m.vocab[key].index
        except KeyError:
            return len(self.m.syn0)

    def vec(self, key):
        try:
            vectors = self.m.vectors
        except AttributeError:
            vectors = self.m.syn0
        try:
            return vectors[self.m.vocab[key].index]
        except KeyError:
            return self.unk


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def eval_emb_metrics(hypotheses: List[List[str]], references: List[List[str]], embedding_array: Embedding):
    emb_hyps = []
    avg_emb_hyps = []
    extreme_emb_hyps = []

    strange_sentence_id = []

    for idx, hyp in enumerate(hypotheses):
        embs = np.array([embedding_array.vec(word) for word in hyp])
        avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
        maxemb = np.max(embs, axis=0)
        minemb = np.min(embs, axis=0)
        extreme_emb = np.array(list(
            map(lambda x, y: x if ((x > y or x < -y) and y > 0) or ((x < y or x > -y) and y < 0) else y, maxemb,
                minemb)))

        emb_hyps.append(embs)
        avg_emb_hyps.append(avg_emb)
        extreme_emb_hyps.append(extreme_emb)

    emb_refs = []
    avg_emb_refs = []
    extreme_emb_refs = []
    for idx, ref in enumerate(references):
        embs = np.array([embedding_array.vec(word) for word in ref])
        avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
        # avg_emb = np.mean(embs,axis=0)
        maxemb = np.max(embs, axis=0)
        minemb = np.min(embs, axis=0)
        extreme_emb = np.array(list(
            map(lambda x, y: x if ((x > y or x < -y) and y > 0) or ((x < y or x > -y) and y < 0) else y, maxemb,
                minemb)))
        emb_refs.append(embs)
        avg_emb_refs.append(avg_emb)
        extreme_emb_refs.append(extreme_emb)

    avg_cos_similarity = np.array([cos_sim(hyp, ref) for hyp, ref in zip(avg_emb_hyps, avg_emb_refs)])
    avg_cos_similarity = avg_cos_similarity.mean()
    extreme_cos_similarity = np.array([cos_sim(hyp, ref) for hyp, ref in zip(extreme_emb_hyps, extreme_emb_refs)])
    extreme_cos_similarity = extreme_cos_similarity.mean()

    scores = []
    for emb_ref, emb_hyp in zip(emb_refs, emb_hyps):
        simi_matrix = cosine_similarity(emb_ref, emb_hyp)
        dir1 = simi_matrix.max(axis=0).mean()
        dir2 = simi_matrix.max(axis=1).mean()
        scores.append((dir1 + dir2) / 2)
    greedy_scores = np.mean(scores)

    print("EmbeddingAverageCosineSimilairty: {0:.6f}".format(avg_cos_similarity))
    print("EmbeddingExtremeCosineSimilairty: {0:.6f}".format(extreme_cos_similarity))
    print("GreedyMatchingScore: {0:.6f}".format(greedy_scores))

    record = {"EmbeddingAverageCosineSimilairty": np.float(avg_cos_similarity),
              "EmbeddingExtremeCosineSimilairty": np.float(extreme_cos_similarity),
              "GreedyMatchingScore": np.float(greedy_scores)}
    return record


def calc_diversity(hypotheses: List[List[str]]):
    unigram_list = list(itertools.chain(*hypotheses))
    total_num_unigram = len(unigram_list)
    unique_num_unigram = len(set(unigram_list))
    bigram_list = []
    for hyp in hypotheses:
        hyp_bigram_list = list(zip(hyp[:-1], hyp[1:]))
        bigram_list += hyp_bigram_list
    total_num_bigram = len(bigram_list)
    unique_num_bigram = len(set(bigram_list))
    dist_1 = unique_num_unigram / total_num_unigram
    dist_2 = unique_num_bigram / total_num_bigram

    return dist_1, dist_2


def eval_diversity_metrics(hypotheses: List[List[str]], references: List[List[str]]):
    h_dist_1, h_dist_2 = calc_diversity(hypotheses)
    r_dist_1, r_dist_2 = calc_diversity(references)
    print("Dist-1 : Preeicted {0:.6f} True {1:.6f}".format(h_dist_1, r_dist_1))
    print("Dist-2 : Predicted {0:.6f} True {1:.6f}".format(h_dist_2, r_dist_2))
    record = {"dist-1": h_dist_1, "dist-2": h_dist_2}
    return record


def calc_bleu(hypotheses: List[List[str]], references: List[List[List[str]]]):
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1., 0., 0., 0.))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(.5, .5, 0., 0.))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(.333, .333, .333, 0.))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(.25, .25, .25, .25))
    avg_bleu = (bleu_1 + bleu_2 + bleu_3 + bleu_4) / 4

    print(f"Bleu_1: {bleu_1:.6f}")
    print(f"Bleu_2: {bleu_2:.6f}")
    print(f"Bleu_3: {bleu_3:.6f}")
    print(f"Bleu_4: {bleu_4:.6f}")
    print(f"average: {avg_bleu:.6f}")

    record = {"Bleu_1": bleu_1, "Bleu_2": bleu_2, "Bleu_3": bleu_3, "Bleu_4": bleu_4, "average_Bleu": avg_bleu}
    return record


def tokenize_raw_text(text_file, lower=True, space_token=False):
    tokenize_text = []
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            if space_token:
                if lower:
                    tokenize_text.append(list(map(str.lower, line.strip().split(" "))))
                else:
                    tokenize_text.append(line.strip().split(" "))
            else:
                if lower:
                    tokenize_text.append(list(map(str.lower, word_tokenize(line.strip()))))
                else:
                    tokenize_text.append(word_tokenize(line.strip()))
    return tokenize_text


def compute_metrics(hyp_file, ref_file, glove_path, lower=False, space_token=False):
    hyps_origin = tokenize_raw_text(hyp_file, lower=lower, space_token=space_token)
    refs_origin = tokenize_raw_text(ref_file, lower=lower, space_token=space_token)
    assert len(hyps_origin) == len(refs_origin)

    # filter empty tokenized sentence
    hyps, refs = [], []
    avg_hyp_length = 0
    for hyp, ref in zip(hyps_origin, refs_origin):
        if len(hyp) == 0:
            continue
        avg_hyp_length += len(hyp)
        hyps.append(hyp)
        refs.append(ref)
    avg_hyp_length = avg_hyp_length / len(hyps)

    print(f"average length: {avg_hyp_length}")
    metric_result = {'average length': avg_hyp_length}
    embd = Embedding(glove_path=glove_path)
    embed_metric = eval_emb_metrics(hyps, refs, embd)
    metric_result.update(embed_metric)
    dist_metric = eval_diversity_metrics(hyps, refs)
    metric_result.update(dist_metric)
    refs = [[ref] for ref in refs]
    bleu_metric = calc_bleu(hyps, refs)
    metric_result.update(bleu_metric)
    return metric_result

# if __name__ == "__main__":
#     glove_path = '/home/liuyuhan/.cache/nlgeval/'
#     m = KeyedVectors.load(os.path.join(glove_path, 'glove.6B.300d.model.bin'), mmap='r')
#     print(sum(m.vectors[0] - m.syn0[0]))
#     ref = [['I', 'love', 'playing', 'in', 'the', 'home']]
#     hyp = ['she', 'really', 'love', 'playing', 'in', 'classroom']
#     ref_text = [" ".join(ref[0])]
#     hyp_text = " ".join(hyp)
#     from nlgeval import compute_individual_metrics
#     print(compute_individual_metrics(ref_text, hyp_text, no_glove=True, no_skipthoughts=True))
#     print(corpus_bleu([ref], [hyp], weights=(1,0,0,0)))
#     print(corpus_bleu([ref], [hyp], weights=(.5,.5)))
#     print(corpus_bleu([ref], [hyp], weights=(1/3,1/3,1/3)))