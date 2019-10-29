''' Convenience functions for handling GPT1 '''
import torch
import sys
import pdb

import pytorch_pretrained_bert as bert
import numpy as np

def load_model(version='openai-gpt'):
    ''' Load GPT1 model and corresponding tokenizer '''
    tokenizer = bert.OpenAIGPTTokenizer.from_pretrained(version)
    model = bert.OpenAIGPTModel.from_pretrained(version)
    model.eval()
    model.to('cuda')

    return model, tokenizer

def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item

def convert_tokens_to_ids(model, tokenizer, tokens, pad=True):
    max_len = model.positions_embed.weight.size(0)
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
    max_len = model.positions_embed.weight.size(0)
    subwords, token_start_idxs = subword_tokenize(tokenizer, tokens)
    subword_ids, mask = convert_tokens_to_ids(model, tokenizer, subwords)
    token_starts = torch.zeros(1, max_len).to(subword_ids)
    token_starts[0, token_start_idxs] = 1
    return subword_ids, mask, token_starts

def encode_sent(model, tokenizer, texts):
    ''' Use tokenizer and model to encode texts '''
    with torch.no_grad():
        encs = {}
        seen = {}
        for text in texts:
            tokenized = tokenizer.tokenize(text)
            indexed = tokenizer.convert_tokens_to_ids(tokenized)
            tokens_tensor = torch.tensor([indexed]).to('cuda')
            enc = model(tokens_tensor)
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
            gpt1_ids, gpt1_mask, gpt1_token_starts = subword_tokenize_to_ids(model, tokenizer, tokens)
            max_length = (gpt1_mask != 0).max(0)[0].nonzero()[-1].item()
            # if max_length < gpt1_ids.shape[1]:
            #       gpt1_ids = gpt1_ids[:, :max_length]
            gpt1_ids = gpt1_ids.to('cuda')
            gpt1_last_layer = model(gpt1_ids)
            gpt1_token_reprs = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(gpt1_last_layer, gpt1_token_starts)]

            if gpt1_token_reprs[0].shape[0] != len(tokens): pdb.set_trace()
            # print(text)
            # print(tokens)
            # print(gpt1_token_reprs[0])
            # print(gpt1_token_reprs[0].shape)
            # print(idx)
            assert(gpt1_token_reprs[0].shape[0] == len(tokens))

            encs[text] = gpt1_token_reprs[0][idx].cpu().view(-1).numpy()
        return encs
