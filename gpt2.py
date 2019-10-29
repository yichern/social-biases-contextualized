''' Convenience functions for handling GPT2 '''
import torch
import sys
import pdb

import pytorch_pretrained_bert as bert
import numpy as np

def load_model(version='gpt2'):
    ''' Load gpt2 model and corresponding tokenizer '''
    tokenizer = bert.GPT2Tokenizer.from_pretrained(version)
    model = bert.GPT2Model.from_pretrained(version)
    model.eval()
    model.to('cuda')

    return model, tokenizer

def load_model_mlm(version='gpt2'):
    ''' Load gpt2 model and corresponding tokenizer '''
    tokenizer = bert.GPT2Tokenizer.from_pretrained(version)
    model = bert.GPT2LMHeadModel.from_pretrained(version)
    model.eval()

    return model, tokenizer

def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item

def convert_tokens_to_ids(model, tokenizer, tokens, pad=True):
    max_len = model.wpe.weight.size(0)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor([token_ids])
    assert ids.size(1) < max_len
    if pad:
        padded_ids = torch.zeros(1, max_len).to(ids)
        padded_ids[0, :ids.size(1)] = ids
        mask = torch.zeros(1, max_len).to(ids)
        mask[0, :ids.size(1)] = 1
        return padded_ids, mask
    else:
        return ids

def subword_tokenize(tokenizer, tokens):
    subwords = list(map(tokenizer.tokenize, tokens))
    subword_lengths = list(map(len, subwords))
    subwords = list(flatten(subwords))
    token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
    return subwords, token_start_idxs

def subword_tokenize_to_ids(model, tokenizer, tokens):
    max_len = model.wpe.weight.size(0)
    subwords, token_start_idxs = subword_tokenize(tokenizer, tokens)
    subword_ids, mask = convert_tokens_to_ids(model, tokenizer, subwords)
    token_starts = torch.zeros(1, max_len).to(subword_ids)
    token_starts[0, token_start_idxs] = 1
    return subword_ids, mask, token_starts

def encode_sent(model, tokenizer, texts):
    ''' Use tokenizer and model to encode texts '''
    with torch.no_grad():
        encs = {}
        for text in texts:
            tokenized = tokenizer.tokenize(text)
            indexed = tokenizer.convert_tokens_to_ids(tokenized)
            tokens_tensor = torch.tensor([indexed]).to('cuda')
            enc, _ = model(tokens_tensor)
            enc = enc[0, -1, :]  # extract the rep of the last input
            # print(enc.size())
            # print(text)
            encs[text] = enc.cpu().numpy()
        return encs

def encode_c_word(model, tokenizer, texts, words):
    ''' Use tokenizer and model to encode texts, but returns contextual representation for word '''
    with torch.no_grad():
        words = list(flatten([word.split() for word in words]))
        encs = {}
        for text in texts:

            # find idx of the word in texts
            idx = None
            tokens = text[:-1].split()
            for i, token in enumerate(tokens):
                if token.lower() in words:
                    idx = i
            if idx is None: pdb.set_trace()
            # print(text)
            # print(tokens)
            # print(words)
            # print(idx)
            assert(idx is not None)

            # get representation for each token
            gpt2_ids, gpt2_mask, gpt2_token_starts = subword_tokenize_to_ids(model, tokenizer, tokens)
            torch.set_printoptions(profile="full")
            # print(gpt2_ids)
            # print(gpt2_mask)
            # print(gpt2_token_starts)
            max_length = (gpt2_mask != 0).max(0)[0].nonzero()[-1].item()
            # print(gpt2_ids.shape[1])
            # print(max_length)
            # if max_length < gpt2_ids.shape[1]:
            #       gpt2_ids = gpt2_ids[:, :max_length]
            # torch.set_printoptions(profile="default")
            gpt2_ids = gpt2_ids.to('cuda')
            gpt2_last_layer, _ = model(gpt2_ids)
            # print(gpt2_last_layer.size())
            # print(gpt2_token_starts.size())
            gpt2_token_reprs = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(gpt2_last_layer, gpt2_token_starts)]

            if gpt2_token_reprs[0].shape[0] != len(tokens): pdb.set_trace()
            # print(text)
            # print(tokens)
            # print(gpt2_token_reprs[0])
            # print(gpt2_token_reprs[0].shape)
            # print(idx)
            assert(gpt2_token_reprs[0].shape[0] == len(tokens))

            encs[text] = gpt2_token_reprs[0][idx].cpu().view(-1).numpy()
        return encs
