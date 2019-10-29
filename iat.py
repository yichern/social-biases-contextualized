# coding=utf-8
import os
import sys
import csv
import math
import random
import glob
import datetime
import argparse
import re
import pdb
import copy

import nltk
import gensim
import spacy
import torch
import numpy as np
import pandas as pd
import logging as log

from tqdm import tqdm
from functools import reduce
from multiprocessing import Pool
from nltk.corpus import stopwords
from pathlib import Path
from collections import Counter
from itertools import count
from csv import DictWriter
from enum import Enum
from allennlp.commands.elmo import ElmoEmbedder

from data import *
from utils import *

import weat
import bow as bow
import bert as bert
import elmo as elmo
import gpt1 as gpt1
import gpt2 as gpt2

class ModelName(Enum):
    BOW = 'bow'
    ELMO = 'elmo'
    BERT = 'bert'
    GPT1 = 'gpt1'
    GPT2 = 'gpt2'

TEST_EXT = '.jsonl'
MODEL_NAMES = [m.value for m in ModelName]
BERT_VERSIONS = ["bert-base-uncased", "bert-large-uncased", "bert-base-cased", "bert-large-cased"]
GPT1_VERSIONS = ["openai-gpt"]
GPT2_VERSIONS = ["gpt2", "gpt2-345"]
WOMAN_RE = re.compile(r'\b(?:woman)\b')
MAN_RE = re.compile(r'\b(?:man)\b')

def test_sort_key(test):
    '''
    Return tuple to be used as a sort key for the specified test name.
    Break test name into pieces consisting of the integers in the name
    and the strings in between them.
    '''
    key = ()
    prev_end = 0
    for match in re.finditer(r'\d+', test):
        key = key + (test[prev_end:match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key

def split_comma_and_check(arg_str, allowed_set, item_type):
    ''' Given a comma-separated string of items,
    split on commas and check if all items are in allowed_set.
    item_type is just for the assert message. '''
    items = arg_str.split(',')
    for item in items:
        if item not in allowed_set:
            raise ValueError("Unknown %s: %s!" % (item_type, item))
    return items

def maybe_make_dir(dirname):
    ''' Maybe make directory '''
    os.makedirs(dirname, exist_ok=True)

def singularize(s):
    if s == 'children':
        return 'child'
    elif s.endswith('s'):
        return s[:-1]
    else:
        return s

def pluralize(s):
    if WOMAN_RE.search(s) is not None:
        return WOMAN_RE.sub('women', s)
    elif MAN_RE.search(s) is not None:
        return MAN_RE.sub('men', s)
    elif s.endswith('y') and s[-2] not in 'aeiou':
        return s[:-1] + 'ies'
    elif s.endswith('ch'):
        return s + 'es'
    elif s.endswith('sh'):
        return s + 'es'
    elif s.endswith('s'):
        return s + 'es'
    else:
        return s + 's'

def augment(words):
    new_words = [word.lower() for word in words]
    for word in words:
        new_words.append(pluralize(word))
        new_words.append(singularize(word))
    return new_words

def main():

    # parser
    parser = argparse.ArgumentParser(
        description='Run specified SEAT tests on specified models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general arguments
    parser.add_argument('--exp', '-e', type=str,
                        help="Run in word or sentence mode", choices=["sent", "word", "c-word"], required=True)
    parser.add_argument('--tests', '-t', type=str,
                        help="WEAT tests to run (a comma-separated list; test files should be in `data_dir` and "
                             "have corresponding names, with extension {}). Default: all tests.".format(TEST_EXT))
    parser.add_argument('--models', '-m', type=str,
                        help="Models to evaluate (a comma-separated list; options: {}). "
                             "Default: all models.".format(','.join(MODEL_NAMES)))
    parser.add_argument('--seed', '-s', type=int, help="Random seed", default=1111)
    parser.add_argument('--log_file', '-l', type=str,
                        help="File to log to")
    parser.add_argument('--results_path', type=str,
                        help="Path where TSV results file will be written")
    parser.add_argument('--ignore_cached_encs', '-i', action='store_true',
                        help="If set, ignore existing encodings and encode from scratch.")
    parser.add_argument('--dont_cache_encs', action='store_true', default=True,
                        help="If set, don't cache encodings to disk.")
    parser.add_argument('--data_dir', '-d', type=str,
                        help="Directory containing examples for each test",
                        default='tests')
    parser.add_argument('--exp_dir', type=str,
                        help="Directory from which to load and save vectors. "
                             "Files should be stored as h5py files.",
                        default='output')
    parser.add_argument('--n_samples', type=int,
                        help="Number of permutation test samples used when estimate p-values (exact test is used if "
                             "there are fewer than this many permutations)",
                        default=100000)
    parser.add_argument('--parametric', action='store_true',
                        help='Use parametric test (normal assumption) to compute p-values.')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Use CPU to encode sentences.')
    parser.add_argument('--glove_path', '-g', type=str,
                        help="File to GloVe vectors in .txt format. "
                             "Required if bow or infersent models are specified.")

    # elmo arguments
    elmo_group = parser.add_argument_group(ModelName.ELMO.value, 'Options for ELMo model')
    elmo_group.add_argument('--time_combine_method', type=str, choices=["max", "mean", "concat", "last"],
                            help="How to combine word representations in ELMo", default="mean")
    elmo_group.add_argument('--layer_combine_method', type=str, choices=["add", "mean", "concat", "last"],
                            help="How to combine layers in ELMo", default="add")

    # bert arguments
    bert_group = parser.add_argument_group(ModelName.BERT.value, 'Options for BERT model')
    bert_group.add_argument('--bert_version', type=str, choices=BERT_VERSIONS,
                            help="Version of BERT to use.", default="bert-large-cased")

    # gpt1 arguments
    gpt1_group = parser.add_argument_group(ModelName.GPT1.value, 'Options for GPT1 model')
    gpt1_group.add_argument('--gpt1_version', type=str, choices=GPT1_VERSIONS,
                            help="Version of OpenAI GPT1 to use.", default="openai-gpt")

    # gpt2 arguments
    gpt2_group = parser.add_argument_group(ModelName.GPT1.value, 'Options for GPT2 model')
    gpt2_group.add_argument('--gpt2_version', type=str, choices=GPT2_VERSIONS,
                            help="Version of OpenAI GPT2 to use.", default="gpt2")

    # parse arguments
    args = parser.parse_args()

    # logging
    log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

    # preliminary handling
    maybe_make_dir(args.exp_dir)
    if args.log_file:
        open(args.log_file, 'w').close()
        fh = log.FileHandler(args.log_file)
        fh.setFormatter(log.Formatter(fmt='%(asctime)s: %(message)s', datefmt="datefmt='%m/%d %I:%M:%S %p"))
        log.getLogger().addHandler(fh)
    if args.seed >= 0:
        log.info('Seeding random number generators with {}'.format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
    log.info("Parsed args: \n%s", args)

    # sort tests
    all_tests = sorted(
        [
            entry[:-len(TEST_EXT)]
            for entry in os.listdir(args.data_dir)
            if not entry.startswith('.') and entry.endswith(TEST_EXT)
        ],
        key=test_sort_key
    )

    # log
    log.debug('Tests found:')
    for test in all_tests:
        log.debug('\t{}'.format(test))

    tests = split_comma_and_check(args.tests, all_tests, "test") if args.tests is not None else all_tests
    log.info('Tests selected:')
    for test in tests:
        log.info('\t{}'.format(test))

    model_name = args.models
    log.info('Model selected:')
    log.info('\t{}'.format(model_name))

    # model options
    if model_name == ModelName.BOW.value:
        model_options = ''
        if args.glove_path is None:
            raise Exception('glove_path must be specified for {} model'.format(model_name))
    elif model_name == ModelName.ELMO.value:
        model_options = 'time_combine={};layer_combine={}'.format(
            args.time_combine_method, args.layer_combine_method)
    elif model_name == ModelName.BERT.value:
        model_options = 'version=' + args.bert_version
    elif model_name == ModelName.GPT1.value:
        model_options = 'version=' + args.gpt1_version
    elif model_name == ModelName.GPT2.value:
        model_options = 'version=' + args.gpt2_version
    else:
        raise ValueError("Model %s not found!" % model_name)

    # load model if needed
    if model_name == ModelName.BOW.value:
        model = None
    elif model_name == ModelName.ELMO.value:
        model = elmo.load_model(device=torch.cuda.current_device())
    elif model_name == ModelName.BERT.value:
        model, tokenizer = bert.load_model(args.bert_version)
    elif model_name == ModelName.GPT1.value:
        model, tokenizer = gpt1.load_model(args.gpt1_version)
    elif model_name == ModelName.GPT2.value:
        model, tokenizer = gpt2.load_model(args.gpt2_version)
    else:
        raise ValueError("Model %s not found!" % model_name)

    # test
    results = []
    for test in tests:
        log.info('Running test {} for model {}'.format(test, model_name))

        # encoder file
        enc_file = os.path.join(args.exp_dir, "%s.%s.h5" % (
            "%s;%s" % (model_name, model_options) if model_options else model_name,
            test))

        # load encoding if possible
        if not args.ignore_cached_encs and os.path.isfile(enc_file):
            log.info("Loading encodings from %s", enc_file)
            encs = load_encodings(enc_file)
            encs_targ1 = encs['targ1']
            encs_targ2 = encs['targ2']
            encs_attr1 = encs['attr1']
            encs_attr2 = encs['attr2']
        else:

            # if word test
            if args.exp == "word" or args.exp == "sent":

                # error checking
                if args.exp == "word" and test[:4] == "sent":
                    raise Exception(f"cannot do word exp on sent iats. exp: {args.exp}, test: {test}")
                if args.exp == "sent" and test[:4] == "weat":
                    raise Exception(f"cannot do sent exp on word iats. exp: {args.exp}, test: {test}")

                # load test data
                encs = load_json(os.path.join(args.data_dir, "%s%s" % (test, TEST_EXT)))

                # load model
                log.info(f"Loading model {model_name}")
                if model_name == ModelName.BOW.value:
                    encs_targ1 = bow.encode(encs["targ1"]["examples"], args.glove_path)
                    encs_targ2 = bow.encode(encs["targ2"]["examples"], args.glove_path)
                    encs_attr1 = bow.encode(encs["attr1"]["examples"], args.glove_path)
                    encs_attr2 = bow.encode(encs["attr2"]["examples"], args.glove_path)

                elif model_name == ModelName.ELMO.value:
                    kwargs = dict(time_combine_method=args.time_combine_method,
                                  layer_combine_method=args.layer_combine_method)
                    encs_targ1 = elmo.encode_sent(model, encs["targ1"]["examples"], **kwargs)
                    encs_targ2 = elmo.encode_sent(model, encs["targ2"]["examples"], **kwargs)
                    encs_attr1 = elmo.encode_sent(model, encs["attr1"]["examples"], **kwargs)
                    encs_attr2 = elmo.encode_sent(model, encs["attr2"]["examples"], **kwargs)

                elif model_name == ModelName.BERT.value:
                    encs_targ1 = bert.encode_sent(model, tokenizer, encs["targ1"]["examples"])
                    encs_targ2 = bert.encode_sent(model, tokenizer, encs["targ2"]["examples"])
                    encs_attr1 = bert.encode_sent(model, tokenizer, encs["attr1"]["examples"])
                    encs_attr2 = bert.encode_sent(model, tokenizer, encs["attr2"]["examples"])

                elif model_name == ModelName.GPT1.value:
                    encs_targ1 = gpt1.encode_sent(model, tokenizer, encs["targ1"]["examples"])
                    encs_targ2 = gpt1.encode_sent(model, tokenizer, encs["targ2"]["examples"])
                    encs_attr1 = gpt1.encode_sent(model, tokenizer, encs["attr1"]["examples"])
                    encs_attr2 = gpt1.encode_sent(model, tokenizer, encs["attr2"]["examples"])

                elif model_name == ModelName.GPT2.value:
                    encs_targ1 = gpt2.encode_sent(model, tokenizer, encs["targ1"]["examples"])
                    encs_targ2 = gpt2.encode_sent(model, tokenizer, encs["targ2"]["examples"])
                    encs_attr1 = gpt2.encode_sent(model, tokenizer, encs["attr1"]["examples"])
                    encs_attr2 = gpt2.encode_sent(model, tokenizer, encs["attr2"]["examples"])

                else:
                    raise ValueError("Model %s not found!" % model_name)

            # elif sentence test
            elif args.exp == "c-word":

                # error checking
                if test[:4] == "weat":
                    raise Exception(f"cannot do c-word exp on word iats. exp: {args.exp}, test: {test}")

                # load test data, both sent and word version
                encs = load_json(os.path.join(args.data_dir, "%s%s" % (test, TEST_EXT)))
                encs_word = load_json(os.path.join(args.data_dir, "%s%s" % (test[5:], TEST_EXT)))
                encs_word["targ1"]["examples"] = augment(encs_word["targ1"]["examples"])
                encs_word["targ2"]["examples"] = augment(encs_word["targ2"]["examples"])
                encs_word["attr1"]["examples"] = augment(encs_word["attr1"]["examples"])
                encs_word["attr2"]["examples"] = augment(encs_word["attr2"]["examples"])

                # load model
                log.info(f"Loading model {model_name}")
                if model_name == ModelName.BOW.value:
                    raise Exception(f"cannot use {model_name} for {args.exp} exp")

                elif model_name == ModelName.ELMO.value:
                    kwargs = dict(time_combine_method=args.time_combine_method,
                                  layer_combine_method=args.layer_combine_method)
                    encs_targ1 = elmo.encode_c_word(model, encs["targ1"]["examples"], encs_word["targ1"]["examples"], args.time_combine_method, args.layer_combine_method)
                    encs_targ2 = elmo.encode_c_word(model, encs["targ2"]["examples"], encs_word["targ2"]["examples"], args.time_combine_method, args.layer_combine_method)
                    encs_attr1 = elmo.encode_c_word(model, encs["attr1"]["examples"], encs_word["attr1"]["examples"], args.time_combine_method, args.layer_combine_method)
                    encs_attr2 = elmo.encode_c_word(model, encs["attr2"]["examples"], encs_word["attr2"]["examples"], args.time_combine_method, args.layer_combine_method)

                elif model_name == ModelName.BERT.value:
                    encs_targ1 = bert.encode_c_word(model, tokenizer, encs["targ1"]["examples"], encs_word["targ1"]["examples"])
                    encs_targ2 = bert.encode_c_word(model, tokenizer, encs["targ2"]["examples"], encs_word["targ2"]["examples"])
                    encs_attr1 = bert.encode_c_word(model, tokenizer, encs["attr1"]["examples"], encs_word["attr1"]["examples"])
                    encs_attr2 = bert.encode_c_word(model, tokenizer, encs["attr2"]["examples"], encs_word["attr2"]["examples"])

                elif model_name == ModelName.GPT1.value:
                    encs_targ1 = gpt1.encode_c_word(model, tokenizer, encs["targ1"]["examples"], encs_word["targ1"]["examples"])
                    encs_targ2 = gpt1.encode_c_word(model, tokenizer, encs["targ2"]["examples"], encs_word["targ2"]["examples"])
                    encs_attr1 = gpt1.encode_c_word(model, tokenizer, encs["attr1"]["examples"], encs_word["attr1"]["examples"])
                    encs_attr2 = gpt1.encode_c_word(model, tokenizer, encs["attr2"]["examples"], encs_word["attr2"]["examples"])

                elif model_name == ModelName.GPT2.value:
                    encs_targ1 = gpt2.encode_c_word(model, tokenizer, encs["targ1"]["examples"], encs_word["targ1"]["examples"])
                    encs_targ2 = gpt2.encode_c_word(model, tokenizer, encs["targ2"]["examples"], encs_word["targ2"]["examples"])
                    encs_attr1 = gpt2.encode_c_word(model, tokenizer, encs["attr1"]["examples"], encs_word["attr1"]["examples"])
                    encs_attr2 = gpt2.encode_c_word(model, tokenizer, encs["attr2"]["examples"], encs_word["attr2"]["examples"])

                else:
                    raise ValueError("Model %s not found!" % model_name)

            encs["targ1"]["encs"] = encs_targ1
            encs["targ2"]["encs"] = encs_targ2
            encs["attr1"]["encs"] = encs_attr1
            encs["attr2"]["encs"] = encs_attr2

            log.info("\tDone!")
            if not args.dont_cache_encs:
                log.info("Saving encodings to %s", enc_file)
                save_encodings(encs, enc_file)

        enc = [e for e in encs["targ1"]['encs'].values()][0]
        d_rep = enc.size if isinstance(enc, np.ndarray) else len(enc)

        # run the test on the encodings
        log.info("Running SEAT...")
        log.info("Representation dimension: {}".format(d_rep))
        esize, pval = weat.run_test(encs, n_samples=args.n_samples, parametric=args.parametric)
        results.append(dict(
            model=model_name,
            options=model_options,
            test=test,
            p_value=pval,
            effect_size=esize,
            num_targ1=len(encs['targ1']['encs']),
            num_targ2=len(encs['targ2']['encs']),
            num_attr1=len(encs['attr1']['encs']),
            num_attr2=len(encs['attr2']['encs'])))

        log.info("Model: %s", model_name)
        log.info('Options: {}'.format(model_options))
        for r in results:
            log.info("\tTest {test}:\tp-val: {p_value:.9f}\tesize: {effect_size:.2f}".format(**r))

    if args.results_path is not None:
        log.info('Writing results to {}'.format(args.results_path))
        with open(args.results_path, 'w') as f:
            writer = DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
            writer.writeheader()
            for r in results:
                writer.writerow(r)

if __name__ == "__main__":
    main()
