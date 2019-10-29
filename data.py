''' Helper functions related to loading and saving data '''
import json
import numpy as np
import h5py
import logging as log
import sys
import os
import re

import deepdish as dd

from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

WEAT_SETS = ["targ1", "targ2", "attr1", "attr2"]
CATEGORY = "category"


def load_json(sent_file):
    ''' Load from json. We expect a certain format later, so do some post processing '''
    log.info("Loading %s..." % sent_file)
    all_data = json.load(open(sent_file, 'r'))
    data = {}
    for k, v in all_data.items():
        examples = v["examples"]
        data[k] = examples
        v["examples"] = examples
    return all_data  # data

def load_wino_mlm(dir, type, case="cased"):
    encs = {"examples": [], "vocab": [], "stereo_labels": []}

    pro_files = [f"pro_stereotyped_type{type}.txt.dev",
                 f"pro_stereotyped_type{type}.txt.test"]
    pro_paths = [Path(os.path.join(dir, file)) for file in pro_files]
    pro_data = []
    for path in pro_paths:
        with path.open(mode='r') as f:
            if case == "uncased":
                pro_data.extend([line.strip().lower().split(' ', 1)[1] for line in f])
            elif case == "cased":
                pro_data.extend([line.strip().split(' ', 1)[1] for line in f])

    anti_files = [f"anti_stereotyped_type{type}.txt.dev",
                  f"anti_stereotyped_type{type}.txt.test"]
    anti_paths = [Path(os.path.join(dir, file)) for file in anti_files]
    anti_data = []
    for path in anti_paths:
        with path.open(mode='r') as f:
            if case == "uncased":
                anti_data.extend([line.strip().lower().split(' ', 1)[1] for line in f])
            elif case == "cased":
                anti_data.extend([line.strip().split(' ', 1)[1] for line in f])

    # process data
    pro_examples = [re.sub(r'\[([\s\w]*)\]', r'\1', sentence) for sentence in pro_data]
    anti_examples = [re.sub(r'\[([\s\w]*)\]', r'\1', sentence) for sentence in anti_data]

    # get stereotypical labels
    pro_stereo_labels = [re.findall(r'\[([\s\w]*)\]', sentence)[1].split(' ', 1)[0] for sentence in pro_data]
    anti_stereo_labels = [re.findall(r'\[([\s\w]*)\]', sentence)[1].split(' ', 1)[0] for sentence in anti_data]

    # get vocab
    vocab = [[word1, word2] for word1, word2 in zip(pro_stereo_labels, anti_stereo_labels)]

    # storage
    encs["examples"] = pro_examples
    encs["vocab"] = vocab
    encs["stereo_labels"] = pro_stereo_labels

    return encs

def load_wino(dir, type, case="uncased", test_size=0.10):
    files = [f"anti_stereotyped_type{type}.txt.dev",
             f"anti_stereotyped_type{type}.txt.test",
             f"pro_stereotyped_type{type}.txt.dev",
             f"pro_stereotyped_type{type}.txt.test"]
    f_occ = "female_occupations.txt"
    m_occ = "male_occupations.txt"

    paths = [Path(os.path.join(dir, file)) for file in files]
    f_occ_path = Path(os.path.join(dir, f_occ))
    m_occ_path = Path(os.path.join(dir, m_occ))

    encs = {"x_train": {}, "y_train": {}, "x_test": {}, "y_test": {}}
    tgts = {}
    data = []
    for path in paths:
        with path.open(mode='r') as f:
            if case == "uncased":
                data.extend([line.strip().lower().split(' ', 1)[1] for line in f])
            elif case == "cased":
                data.extend([line.strip().split(' ', 1)[1] for line in f])

    # split to training and test set
    seed = 123
    train, test = train_test_split(data, test_size=test_size, random_state=seed)

    # process train and test
    x_train = [re.sub(r'\[([\s\w]*)\]', r'\1', sentence) for sentence in train]
    x_test = [re.sub(r'\[([\s\w]*)\]', r'\1', sentence) for sentence in test]

    # get targets
    tgts_x_train = [re.findall(r'\[([\s\w]*)\]', sentence)[0].split(' ', 1)[1] for sentence in train]
    tgts_x_test = [re.findall(r'\[([\s\w]*)\]', sentence)[0].split(' ', 1)[1] for sentence in test]

    # get labels
    with f_occ_path.open(mode='r') as f:
        f_occs = [line.strip().lower() for line in f]
    with m_occ_path.open(mode='r') as f:
        m_occs = [line.strip().lower() for line in f]
    y_train = ["male" if tgt in m_occs else "female" for tgt in tgts_x_train ]
    y_test = ["male" if tgt in m_occs else "female" for tgt in tgts_x_test ]

    # print(f"x_train: {len(x_train)}")
    # print(f"x_test: {len(x_test)}")
    # print(f"y_train: {Counter(y_train)}")
    # print(f"y_test: {Counter(y_test)}")
    # print(x_train[0])
    # print(tgts_x_train[0])
    # print(y_train[0])

    # storage
    encs["x_train"]["examples"] = x_train
    encs["y_train"]["examples"] = y_train
    encs["x_test"]["examples"] = x_test
    encs["y_test"]["examples"] = y_test
    tgts["x_train"] = tgts_x_train
    tgts["x_test"] = tgts_x_test

    return encs, tgts

def load_encodings(enc_file):
    ''' Load cached vectors from a model. '''
    encs = dict()
    with h5py.File(enc_file, 'r') as enc_fh:
        for split_name, split in enc_fh.items():
            split_d, split_exs = {}, {}
            for ex, enc in split.items():
                if ex == CATEGORY:
                    split_d[ex] = enc.value
                else:
                    split_exs[ex] = enc[:]
            split_d["encs"] = split_exs
            encs[split_name] = split_d
    return encs

def save_encodings(encodings, enc_file):
    ''' Save encodings to file '''
    with h5py.File(enc_file, 'w') as enc_fh:
        for split_name, split_d in encodings.items():
            split = enc_fh.create_group(split_name)
            split[CATEGORY] = split_d["category"]
            for ex, enc in split_d["encs"].items():
                split[ex] = enc

def load_encodings_wino(enc_file):
    ''' Load cached vectors from a model. '''
    encs = dd.io.load(enc_file)
    return encs

def save_encodings_wino(encodings, enc_file):
    ''' Save encodings to file '''
    dd.io.save(enc_file, encodings)

def load_jiant_encodings(enc_file, n_header=1, is_openai=False):
    ''' Load a dumb tsv format of jiant encodings.
    This is really brittle and makes a lot of assumptions about ordering. '''
    encs = []
    last_cat = None
    with open(enc_file, 'r') as enc_fh:
        for _ in range(n_header):
            enc_fh.readline()  # header
        for row in enc_fh:
            idx, category, string, enc = row.strip().split('\t')
            if is_openai:
                string = " ".join([w.rstrip("</w>") for w in string])
            enc = [float(n) for n in enc[1:-1].split(',')]
            # encs[category][string] = np.array(enc)
            if last_cat is None or last_cat != category:
                # encs.append([np.array(enc)])
                encs.append({string: np.array(enc)})
            else:
                # encs[-1].append(np.array(enc))
                encs[-1][string] = np.array(enc)
            last_cat = category
            # encs[category].append(np.array(enc))

    return encs
