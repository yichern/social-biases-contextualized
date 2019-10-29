# coding=utf-8
import os
import sys
import csv
import math
import random
import glob
import datetime
import argparse
from itertools import count
import re
import json

import nltk
import gensim
import numpy as np
import pandas as pd
import spacy

from tqdm import tqdm
from functools import reduce
from multiprocessing import Pool
from nltk.corpus import stopwords
from pathlib import Path
from collections import Counter

from utils import *

def process_wiki_file(wiki_file):
    chars = ['\n']
    results = []
    with open(wiki_file, encoding='utf-8') as f:
        content = f.read()
        articles = splitkeepsep(content,'<doc id=')
        for article in articles:
            article = remove_special_chars(remove_html_tags(article), chars)

            # doc = nlp(article)
            # proc_sentences = [clean_string(sentence.text) for sentence in list(doc.sents)]

            sentences = nltk.sent_tokenize(article)
            proc_sentences = [clean_string(sentence) for sentence in sentences]

            results.extend(proc_sentences)
    return results

# # multiprocessing example
# import os
# import multiprocessing
#
# tld = [os.path.join("/", f) for f in os.walk("/").next()[1]]
# manager = multiprocessing.Manager()
#
# files = manager.list()
# def get_files(x):
#     for root, dir, file in os.walk(x):
#         for name in file:
#             files.append(os.path.join(root, name))
#
# pool = multiprocessing.Pool(processes=len(tld)) # Instantiate the pool here
#
# pool.map(get_files, [x for x in tld])
# pool.close()
# pool.join()
# print len(files)

def load_1bword(path):
    paths = [Path(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    sentences = []
    for path in paths:
        with path.open(mode='r') as f:
            sentences.extend([line.strip() for line in f])
    return sentences

def load_wikipedia(path):
    paths = []
    for filename in glob.iglob(f"{path}/*/*", recursive=True):
        paths.append(Path(os.path.join(path, filename)))

    # global nlp
    # nlp = spacy.load('en')

    print(f"starting loading, {datetime.datetime.now()}")
    sys.stdout.flush()
    workers = 16
    with Pool(processes=workers) as pool:
        results = pool.map(process_wiki_file, paths)
    print(f"finished loading, {datetime.datetime.now()}")
    sys.stdout.flush()

    print(f"starting acc, {datetime.datetime.now()}")
    sys.stdout.flush()
    final = sum(results, [])
    print(f"finished acc, {datetime.datetime.now()}")
    sys.stdout.flush()

    return final

def load_bookcorpus(path):
    with path.open(mode='r') as f:
        sentences = [line.strip() for line in f]
    return sentences

def load_webtext(path):
    with path.open(mode='r') as f:
        data = []
        sentences = []
        for line in f:
            data.append(json.loads(line))
        for datum in data:
            texts = datum["text"].split("\n\n")
            for text in texts:
                text = text.split(". ")
                if len(text) == 1:
                    sentences.append(text[0])
                else:
                    sentences.append(text[1])
        return sentences

def update_count_para(sentence):
    counts = {"male_pronouns": 0,
              "female_pronouns": 0,
              "neutral_pronouns": 0,
              "male_pro_stereo": 0,
              "male_anti_stereo": 0,
              "female_pro_stereo": 0,
              "female_anti_stereo": 0,
              "male_neutral": 0,
              "female_neutral": 0}
    counts = Counter(counts)
    tokens = sentence.lower().split()
    male_occ = False
    female_occ = False
    neutral_occ = False

    for m_pn in m_pns:
        if m_pn in tokens:
            male_occ = True
            counts["male_pronouns"] += 1
            break

    for f_pn in f_pns:
        if f_pn in tokens:
            female_occ = True
            counts["female_pronouns"] += 1
            break

    for n_pn in n_pns:
        if n_pn in tokens:
            neutral_occ = True
            counts["neutral_pronouns"] += 1
            break

    for m_tgt in m_tgts:
        if m_tgt in tokens and male_occ:
            counts["male_pro_stereo"] += 1
        if m_tgt in tokens and female_occ:
            counts["female_anti_stereo"] += 1
        if m_tgt in tokens and neutral_occ:
            counts["male_neutral"] += 1

    for f_tgt in f_tgts:
        if f_tgt in tokens and female_occ:
            counts["female_pro_stereo"] += 1
        if f_tgt in tokens and male_occ:
            counts["male_anti_stereo"] += 1
        if f_tgt in tokens and neutral_occ:
            counts["female_neutral"] += 1

    return counts

def load_tgts(path):
    with path.open(mode='r') as f:
        tgts = [line.strip().lower() for line in f]
    return tgts

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        choices=['1bword', 'wikipedia', 'bookcorpus', 'webtext'],
                        help="Specifies the dataset to check statistics for")

    parser.add_argument("--data_path",
                        type=str,
                        required=True,
                        help="Relative directory where dataset is stored")

    parser.add_argument("--f_att_tgt",
                        type=str,
                        required=True,
                        help="Target words that stereotypically belong to female attribute")

    parser.add_argument("--m_att_tgt",
                        type=str,
                        required=True,
                        help="Target words that stereotypically belong to male attribute")

    args = parser.parse_args()

    # load dataset
    if args.dataset == "1bword":
        sentences = load_1bword(Path(args.data_path))
    elif args.dataset == "wikipedia":
        sentences = load_wikipedia(Path(args.data_path))
    elif args.dataset == "bookcorpus":
        sentences = load_bookcorpus(Path(args.data_path))
    elif args.dataset == "webtext":
        sentences = load_webtext(Path(args.data_path))
    else:
        raise Exception(f"error: the following datasets must be specified: 1bword, wikipedia, bookcorpus, webtext")
    print(f"Number of sentences: {len(sentences)}")
    print(f"First sentence: {sentences[0]}")
    sys.stdout.flush()

    # global
    global m_pns
    global f_pns
    global n_pns
    global m_tgts
    global f_tgts

    # m/f pronouns
    m_pns = ["he", "him", "his"]
    f_pns = ["she", "her", "hers"]
    n_pns = ["they", "them", "their", "theirs"]

    # m/f tgt words
    m_tgts = load_tgts(Path(args.m_att_tgt))
    f_tgts = load_tgts(Path(args.f_att_tgt))

    # update counts for each sentence
    print(f"starting count, {datetime.datetime.now()}")
    sys.stdout.flush()
    workers = 16
    with Pool(processes=workers) as pool:
        results = pool.map(update_count_para, sentences)
    print(f"finished count, {datetime.datetime.now()}")
    sys.stdout.flush()

    # accumulate counts
    print(f"starting acc, {datetime.datetime.now()}")
    sys.stdout.flush()
    master_count = sum(results, Counter())
    print(f"finished acc, {datetime.datetime.now()}")
    sys.stdout.flush()

    # print counts
    for key, value in master_count.items():
        print(f"{key}: {value}")

if __name__ == '__main__':

    # # test
    # counts = {"male_pronouns": 0,
    #           "female_pronouns": 0,
    #           "male_pro_stereo": 0,
    #           "male_anti_stereo": 0,
    #           "female_pro_stereo": 0,
    #           "female_anti_stereo": 0}
    # sentence = "her developer was his CEO , but his designer and her CEO was hers . "
    # m_tgts = ["developer", "ceo"]
    # f_tgts = ["designer"]
    # m_pns = ["he", "him", "his"]
    # f_pns = ["she", "her", "hers"]
    # update_count(counts, sentence, m_pns, f_pns, m_tgts, f_tgts)
    # for key, value in counts.items():
    #     print(f"{key}: {value}")

    # # test wiki process
    # global nlp
    # nlp = spacy.load('en')
    # print(f"starting acc, {datetime.datetime.now()}")
    # sys.stdout.flush()
    # sentences = process_wiki_file(Path("wiki_00"))
    # print(f"finished acc, {datetime.datetime.now()}")
    # sys.stdout.flush()
    # print(len(sentences))
    # for sentence in sentences:
    #     print(sentence)

    # main
    main()
