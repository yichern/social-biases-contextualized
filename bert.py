''' Convenience functions for handling BERT '''
import torch
import sys
import pdb

import pytorch_pretrained_bert as bert
import numpy as np

def load_model(version='bert-large-uncased'):
    ''' Load BERT model and corresponding tokenizer '''
    tokenizer = bert.BertTokenizer.from_pretrained(version)
    model = bert.BertModel.from_pretrained(version)
    model.eval()
    model.to('cuda')

    return model, tokenizer

def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item

def convert_tokens_to_ids(model, tokenizer, tokens, pad=True):
    max_len = model.embeddings.position_embeddings.weight.size(0)
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
    subwords = ["[CLS]"] + list(flatten(subwords)) + ["[SEP]"]
    token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
    return subwords, token_start_idxs

def subword_tokenize_to_ids(model, tokenizer, tokens):
    max_len = model.embeddings.position_embeddings.weight.size(0)
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
            tokenized = ["[CLS]"] + list(flatten(tokenized)) + ["[SEP]"]
            indexed = tokenizer.convert_tokens_to_ids(tokenized)
            segment_idxs = [0] * len(tokenized)
            tokens_tensor = torch.tensor([indexed]).to('cuda')
            segments_tensor = torch.tensor([segment_idxs]).to('cuda')
            enc, _ = model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)

            enc = enc[:, 0, :]  # extract the last rep of the first input
            encs[text] = enc.cpu().detach().view(-1).numpy()
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
            # print(text)
            # print(tokens)
            # print(words)
            # print(idx)
            if idx is None: pdb.set_trace()
            assert(idx is not None)

            # get representation for each token
            bert_ids, bert_mask, bert_token_starts = subword_tokenize_to_ids(model, tokenizer, tokens)
            max_length = (bert_mask != 0).max(0)[0].nonzero()[-1].item()
            if max_length < bert_ids.shape[1]:
                  bert_ids = bert_ids[:, :max_length]
                  bert_mask = bert_mask[:, :max_length]
            segment_ids = torch.zeros_like(bert_mask)  # dummy segment IDs, since we only have one sentence
            bert_ids = bert_ids.to('cuda')
            segment_ids = segment_ids.to('cuda')
            bert_last_layer = model(bert_ids, segment_ids)[0][-1]
            # print(bert_last_layer.size())
            # print(bert_token_starts.size())
            bert_token_reprs = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(bert_last_layer, bert_token_starts)]
            if bert_token_reprs[0].shape[0] != len(tokens): pdb.set_trace()
            # print(text)
            # print(tokens)
            # print(bert_token_reprs[0])
            # print(bert_token_reprs[0].shape)
            # print(idx)
            assert(bert_token_reprs[0].shape[0] == len(tokens))

            encs[text] = bert_token_reprs[0][idx].cpu().view(-1).numpy()
        return encs


def encode_mlm(model, tokenizer, texts, vocabs, labels):

    with torch.no_grad():
        results = []
        for text, label, vocab in zip(texts, labels, vocabs):

            # tokenize text
            tokens = text[:-1].split()
            bert_tokens, bert_token_starts = subword_tokenize(tokenizer, tokens)
            idx = bert_tokens.index(label)

            # mask tokens
            bert_tokens[idx] = '[MASK]'

            # convert to vocabulary indices
            bert_ids, bert_mask = convert_tokens_to_ids(model, tokenizer, bert_tokens)

            # variables
            max_length = (bert_mask != 0).max(0)[0].nonzero()[-1].item()
            if max_length < bert_ids.shape[1]:
                  bert_ids = bert_ids[:, :max_length]
                  bert_mask = bert_mask[:, :max_length]
            segment_ids = torch.zeros_like(bert_mask)  # dummy segment IDs, since we only have one sentence

            # get predictions
            predictions = model(bert_ids, segment_ids)

            # get probabilities
            result_dict = {}
            softmax = torch.nn.Softmax(dim=1)
            vocab_dist = torch.unsqueeze(predictions[0,idx],0)
            vocab_dist = softmax(vocab_dist).cpu().numpy()[0]
            for key in vocab:
                key_tok = tokenizer.tokenize(key)
                key_idx = tokenizer.convert_tokens_to_ids(key_tok)[0]
                result_dict[key] = vocab_dist[key_idx]
            results.append(result_dict)
        return results

def encode_sent_wino(model, tokenizer, texts, layers):
    ''' Use tokenizer and model to encode texts '''
    with torch.no_grad():
        encs = {layer: [] for layer in layers}
        for text in texts:
            tokenized = tokenizer.tokenize(text)
            indexed = tokenizer.convert_tokens_to_ids(tokenized)
            segment_idxs = [0] * len(tokenized)
            tokens_tensor = torch.tensor([indexed])
            segments_tensor = torch.tensor([segment_idxs])
            all_enc, _ = model(tokens_tensor, segments_tensor, output_all_encoded_layers=True)

            for layer in layers:
                enc = all_enc[layer]
                enc = enc[:, 0, :]  # extract the last rep of the first input
                encs[layer].append(enc.view(-1).numpy())
        return encs

def encode_word_context_wino(model, tokenizer, texts, words, layers):
    ''' Use tokenizer and model to encode texts, but returns contextual representation for word '''
    with torch.no_grad():
        encs = {layer: [] for layer in layers}
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
            # print(text)
            # print(word)
            # print(tokens)
            # print(idx)
            assert(idx is not None)

            # get representation for each token
            bert_ids, bert_mask, bert_token_starts = subword_tokenize_to_ids(model, tokenizer, tokens)
            max_length = (bert_mask != 0).max(0)[0].nonzero()[-1].item()
            if max_length < bert_ids.shape[1]:
                  bert_ids = bert_ids[:, :max_length]
                  bert_mask = bert_mask[:, :max_length]
            segment_ids = torch.zeros_like(bert_mask)  # dummy segment IDs, since we only have one sentence
            bert_all_layers = model(bert_ids, segment_ids)[0]

            for layer in layers:
                bert_single_layer = bert_all_layers[layer]
                bert_token_reprs = [layer_[starts.nonzero().squeeze(1)] for layer_, starts in zip(bert_single_layer, bert_token_starts)]


                # print(bert_token_reprs[0])
                # print(bert_token_reprs[0].shape)

                assert(bert_token_reprs[0].shape[0] == len(tokens))
                # sys.exit(0)

                encs[layer].append(bert_token_reprs[0][idx].view(-1).numpy())
        return encs
