import numpy as np
import torch
from allennlp.commands.elmo import ElmoEmbedder
import pdb

def load_model(device=-1):
    ''' Load ELMoEmbedder with CUDA '''
    model = ElmoEmbedder(cuda_device=device)
    return model

def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item

def encode_sent(model, sents, time_combine_method="mean", layer_combine_method="add"):
    """ Load ELMo and encode sents """
    vecs = {}
    for sent in sents:
        vec_seq = model.embed_sentence(sent)
        if time_combine_method == "max":
            vec = vec_seq.max(axis=1)
        elif time_combine_method == "mean":
            vec = vec_seq.mean(axis=1)
        elif time_combine_method == "concat":
            vec = np.concatenate(vec_seq, axis=1)
        elif time_combine_method == "last":
            vec = vec_seq[:, -1]
        else:
            raise NotImplementedError

        if layer_combine_method == "add":
            vec = vec.sum(axis=0)
        elif layer_combine_method == "mean":
            vec = vec.mean(axis=0)
        elif layer_combine_method == "concat":
            vec = np.concatenate(vec, axis=0)
        elif layer_combine_method == "last":
            vec = vec[-1]
        else:
            raise NotImplementedError
        vecs[' '.join(sent)] = vec
    return vecs

def encode_c_word(model, texts, words, time_combine_method="mean", layer_combine_method="add"):
    """ Load ELMo and encode sents """
    words = list(flatten([word.split() for word in words]))
    vecs = {}
    for text in texts:

        # find idx of the word in texts
        idx = None
        tokens = text[:-1].split()
        for i, token in enumerate(tokens):
            if token.lower() in words:
                idx = i
        if idx is None: pdb.set_trace()
        # print(text)
        # print(word)
        # print(tokens)
        # print(idx)
        assert(idx is not None)

        # get elmo rep of each token
        vec_seq = model.embed_sentence(text)

        # get elmo rep for tokene we want
        vec = vec_seq[:, idx]

        # combine layers
        if layer_combine_method == "add":
            vec = vec.sum(axis=0)
        elif layer_combine_method == "mean":
            vec = vec.mean(axis=0)
        elif layer_combine_method == "concat":
            vec = np.concatenate(vec, axis=0)
        elif layer_combine_method == "last":
            vec = vec[-1]
        else:
            raise NotImplementedError
        vecs[' '.join(text)] = vec
    return vecs

def encode_sent_wino(texts, layers, time_combine_method="mean", layer_combine_method="add"):
    """ Load ELMo and encode sents """
    elmo = ElmoEmbedder()

    if layers is None:

        vecs = []
        for text in texts:
            vec_seq = elmo.embed_sentence(text)
            if time_combine_method == "max":
                vec = vec_seq.max(axis=1)
            elif time_combine_method == "mean":
                vec = vec_seq.mean(axis=1)
            elif time_combine_method == "concat":
                vec = np.concatenate(vec_seq, axis=1)
            elif time_combine_method == "last":
                vec = vec_seq[:, -1]
            else:
                raise NotImplementedError

            if layer_combine_method == "add":
                vec = vec.sum(axis=0)
            elif layer_combine_method == "mean":
                vec = vec.mean(axis=0)
            elif layer_combine_method == "concat":
                vec = np.concatenate(vec, axis=0)
            elif layer_combine_method == "last":
                vec = vec[-1]
            else:
                raise NotImplementedError
            vecs.append(vec)
        return vecs

    else:
        vecs = {i: [] for i, layer in enumerate(layers)}
        for text in texts:
            vec_seq = elmo.embed_sentence(text)
            if time_combine_method == "max":
                vec = vec_seq.max(axis=1)
            elif time_combine_method == "mean":
                vec = vec_seq.mean(axis=1)
            elif time_combine_method == "concat":
                vec = np.concatenate(vec_seq, axis=1)
            elif time_combine_method == "last":
                vec = vec_seq[:, -1]
            else:
                raise NotImplementedError

            for layer in layers:
                vecs[layer].append(vec[layer])
        return vecs

def encode_word_context_wino(texts, words, layers, time_combine_method="mean", layer_combine_method="add"):
    """ Load ELMo and encode sents """
    elmo = ElmoEmbedder()

    if layers is None:

        vecs = []

        for text, word in zip(texts, words):

            # find idx of the word in texts
            idx = None
            tokens = text[:-1].split()
            for i, token in enumerate(tokens):
                if token.lower() == word.lower():
                    idx = i
                if len(word.split(' ', 1)) != 1 :
                    if token.lower() == word.lower().split(' ', 1)[1]:
                        idx = i
            assert(idx is not None)

            # get elmo rep of each token
            vec_seq = elmo.embed_sentence(text)

            # geet elmo rep for tokene we want
            vec = vec_seq[:, idx]

            # combine layers
            if layer_combine_method == "add":
                vec = vec.sum(axis=0)
            elif layer_combine_method == "mean":
                vec = vec.mean(axis=0)
            elif layer_combine_method == "concat":
                vec = np.concatenate(vec, axis=0)
            elif layer_combine_method == "last":
                vec = vec[-1]
            else:
                raise NotImplementedError

            vecs.append(vec)
        return vecs

    else:

        vecs = {i: [] for i, layer in enumerate(layers)}

        for text, word in zip(texts, words):

            # find idx of the word in texts
            idx = None
            tokens = text[:-1].split()
            for i, token in enumerate(tokens):
                if token.lower() == word.lower():
                    idx = i
                if len(word.split(' ', 1)) != 1 :
                    if token.lower() == word.lower().split(' ', 1)[1]:
                        idx = i
            assert(idx is not None)

            # get elmo rep of each token
            vec_seq = elmo.embed_sentence(text)

            # geet elmo rep for tokene we want
            vec = vec_seq[:, idx]

            for layer in layers:
                vecs[layer].append(vec[layer])
        return vecs
