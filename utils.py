import os
import json
import requests
from scipy.stats import pearsonr
import numpy as np
import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re

# part 1
def datamuse(query, max_num):
    url = 'https://api.datamuse.com/words?ml=%s&max=%d' % (query, max_num)

    response = requests.get(url)
    results = response.text

    results = json.loads(results)

    words = [a['word'] for a in results]

    return words

# part 2
def jaccard_similarity(a, b):
    overlap = [element for element in a if element in b]
    return 1.0 * len(overlap) / ((len(a) + len(b)) - len(overlap))

def simple_word_similarity(w1, w2, max_num=30):
    assert(isinstance(w1, str)), 'the first input parameter is not a string variable'
    assert(isinstance(w2, str)), 'the second input parameter is not a string variable'

    words_1 = datamuse(w1, max_num=max_num)
    words_2 = datamuse(w2, max_num=max_num)

    return jaccard_similarity(words_1, words_2)

# part 3: function to calculate pearson coeffficients
def pearson_correlation(data1, data2):
    corr, _ = pearsonr(data1, data2)
    print('Pearsons correlation: %.3f' % corr)
    return corr

# part 4
def simple_sentence_similarity(s1, s2):
    s1 = re.sub('[^a-zA-Z]', ' ', s1 )
    s1 = re.sub('\s+', ' ', s1 )
    s2 = re.sub('[^a-zA-Z]', ' ', s2 )
    s2 = re.sub('\s+', ' ', s2 )
    s1 = s1.strip().split()
    s2 = s2.strip().split()
    overlap_1 = None
    for w in s1:
        words = datamuse(w, 500)
        if not overlap_1:
            overlap_1 = words
        else:
            overlap_1 = [element for element in words if element in overlap_1]
    overlap_1.extend(s1)

    overlap_2 = None
    for w in s2:
        words = datamuse(w, 500)
        if not overlap_2:
            overlap_2 = words
        else:
            overlap_2 = [element for element in words if element in overlap_2]
    overlap_2.extend(s2)
    return jaccard_similarity(overlap_1, overlap_2)

# Using datamuse
def words_similarity_dataset(Dataset,max_num=30):
    sim_list = []
    for i, word_pair in enumerate(Dataset):
        print('%d/%d th pair' % (i + 1, len(Dataset)))
        w1 = word_pair[0]
        w2 = word_pair[1]
        similarity = simple_word_similarity(w1, w2, max_num=max_num)
        sim_list.append(similarity)
    return sim_list

def sentence_similarity_dataset(Dataset):
    sim_list = []
    for i, sentence_pair in enumerate(Dataset):
        print('%d/%d th pair' % (i + 1, len(Dataset)))
        similarity = simple_sentence_similarity(sentence_pair[0], sentence_pair[1])
        sim_list.append(similarity)
    return sim_list


# Using Word2Vec, Glove, FastText
def model_sentence_similarity(s1, s2, model):
    s1 = re.sub('[^a-zA-Z]', ' ', s1 )
    s1 = re.sub('\s+', ' ', s1 )
    s2 = re.sub('[^a-zA-Z]', ' ', s2 )
    s2 = re.sub('\s+', ' ', s2 )
    s1 = s1.strip().split()
    s2 = s2.strip().split()
    vec_1 = None
    for w in s1:
        if vec_1 is None:
            try:
                vec_1 = model[w]
            except Exception as e:
                pass
        else:
            try:
                vec_1 += model[w]
            except Exception as e:
                pass
    vec_1 = vec_1 / len(s1)

    vec_2 = None
    for w in s2:
        if vec_2 is None:
            try:
                vec_2 = model[w]
            except Exception as e:
                pass
        else:
            try:
                vec_2 += model[w]
            except Exception as e:
                pass
    vec_2 = vec_2 / len(s2)
    return cosine_similarity(vec_1.reshape(1, -1), vec_2.reshape(1, -1))

def sentence_similarity_dataset_model(Dataset, model):
    sim_list = []
    for i, sentence_pair in enumerate(Dataset):
        print('%d/%d th pair' % (i + 1, len(Dataset)))
        similarity = model_sentence_similarity(sentence_pair[0], sentence_pair[1], model)
        sim_list.append(similarity)
    return sim_list


# Using Yago
def yago_word_similarity(w1, w2, sim):
    concepts_1 = sim.word2yago(w1)
    concepts_2 = sim.word2yago(w2)
    if concepts_1 and concepts_2:
        return np.array([sim.yago_similarity(c1, c2) for c1 in concepts_1 for c2 in concepts_2]).max()
    else:
        return 0

def yago_sentence_similarity(s1, s2, sim):
    s1 = re.sub('[^a-zA-Z]', ' ', s1 )
    s1 = re.sub('\s+', ' ', s1 )
    s2 = re.sub('[^a-zA-Z]', ' ', s2 )
    s2 = re.sub('\s+', ' ', s2 )
    s1 = s1.strip().split()
    s2 = s2.strip().split()
    return np.array([yago_word_similarity(w1, w2, sim) for w1 in s1 for w2 in s2]).mean()

def sentence_similarity_dataset_yago(Dataset, sim):
    sim_list = []
    for i, sentence_pair in enumerate(Dataset):
        print('%d/%d th pair' % (i + 1, len(Dataset)))
        similarity = yago_sentence_similarity(sentence_pair[0], sentence_pair[1], sim)
        sim_list.append(similarity)
    return sim_list

