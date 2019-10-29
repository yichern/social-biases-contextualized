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

def _remove_non_printed_chars(string):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ]')
    return reg.sub(' ', string)

def _trim_string(string):
    # remove extra spaces, remove trailing spaces, lower the case
    return re.sub('\s+',' ',string).strip().lower()

def clean_string(string):
    string = _remove_non_printed_chars(string)
    string = _trim_string(string)
    return string

def splitkeepsep(s, sep):
    cleaned = []
    s = re.split("(%s)" % re.escape(sep), s)
    for _ in s:
        if _!='' and _!=sep:
            cleaned.append(sep+_)
    return cleaned

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_special_chars(text,char_list):
    for char in char_list:
        text=text.replace(char,'')
    return text.replace(u'\xa0', u' ')
