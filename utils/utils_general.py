import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import random


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


if args['dataset'] == 'kvr':
    domains = {'navigate': 0, 'weather': 1, 'schedule': 2}
elif args['dataset'] == 'woz':
    domains = {'restaurant': 0, 'attraction': 1, 'hotel': 2}
elif args['dataset'] == 'car':
    domains = {'restaurant': 0}


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK', SEP_token: 'SEP'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, src_word2id, trg_word2id, lang):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.lang = lang

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        context_arr = self.data_info['context_arr'][index]
        context_arr = self.preprocess(context_arr, self.src_word2id, trg=False)
        response = self.data_info['response'][index]
        response = self.preprocess(response, self.trg_word2id)
        ptr_index = torch.Tensor(self.data_info['ptr_index'][index])
        selector_index = torch.Tensor(self.data_info['selector_index'][index])
        ent_selector_index = torch.Tensor(self.data_info['ent_selector_index'][index])
        conv_arr_words = self.data_info['conv_arr'][index]
        conv_arr = self.preprocess(conv_arr_words, self.src_word2id, trg=False)
        kb_arr = self.data_info['kb_arr'][index]
        kb_arr = self.preprocess(kb_arr, self.src_word2id, trg=False)
        sketch_response = self.data_info['sketch_response'][index]
        sketch_response = self.preprocess(sketch_response, self.trg_word2id)

        context_word_lengths = torch.Tensor(self.data_info['context_word_lengths'][index])
        conv_word_lengths = torch.Tensor(self.data_info['conv_word_lengths'][index])
        conv_ent_mask = torch.Tensor(self.data_info['conv_ent_mask'][index])
        # kb_txt = self.data_info['kb_txt'][index]
        # kb_txt = self.preprocess(kb_txt, self.src_word2id)[:-1]
        # conv_u_tf = self.data_info['conv_u_tf'][index]
        # conv_u_tf = self.preprocess(conv_u_tf, self.src_word2id)[:-1]
        conv_u_words = self.data_info['conv_u'][index]
        conv_u = self.preprocess(conv_u_words, self.src_word2id, trg=False)

        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        # additional plain information
        data_info['context_arr_plain'] = self.data_info['context_arr'][index]
        data_info['response_plain'] = self.data_info['response'][index]
        data_info['dialog_template_plain'] = " ".join([item[0] for item in conv_u_words])
        data_info['dialog_plain'] = " ".join([item[0] for item in conv_arr_words])
        data_info['gold_sketch_response'] = self.data_info['sketch_response'][index]
        data_info['kb_arr_plain'] = self.data_info['kb_arr'][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        story = torch.Tensor(story)
        return story

    def collate_fn(self, data):
        def merge(sequences, story_dim):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            if (story_dim):
                padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    if len(seq) != 0:
                        padded_seqs[i, :end, :] = seq[:end]
            else:
                padded_seqs = torch.ones(len(sequences), max_len).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        def merge_index(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths
        def merge_word_lens(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths), MEM_TOKEN_SIZE).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                if len(seq) != 0:
                    padded_seqs[i, :end, :] = seq[:end]
            return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x['conv_arr']), reverse=True)
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences
        context_arr, context_arr_lengths = merge(item_info['context_arr'], True)
        response, response_lengths = merge(item_info['response'], False)
        selector_index, _ = merge_index(item_info['selector_index'])
        ent_selector_index, _ = merge_index(item_info['ent_selector_index'])
        conv_ent_mask, _ = merge_index(item_info['conv_ent_mask'])
        ptr_index, _ = merge(item_info['ptr_index'], False)
        conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], True)
        sketch_response, _ = merge(item_info['sketch_response'], False)
        kb_arr, kb_arr_lengths = merge(item_info['kb_arr'], True)

        # kb_txt, _ = merge(item_info['kb_txt'], False)
        # conv_u_tf, _ = merge(item_info['conv_u_tf'], False)
        conv_u, conv_u_lengths = merge(item_info['conv_u'], True)
        context_word_lengths, _ = merge_word_lens(item_info['context_word_lengths'])
        conv_word_lengths, _ = merge_word_lens(item_info['conv_word_lengths'])

        max_seq_len = conv_arr.size(1)
        label_arr = _cuda(torch.Tensor([domains[label] for label in item_info['domain']]).long().unsqueeze(-1))
        # convert to contiguous and cuda
        context_arr = _cuda(context_arr.contiguous())
        response = _cuda(response.contiguous())
        # kb_txt = _cuda(kb_txt.contiguous())
        # conv_u_tf = _cuda(conv_u_tf.contiguous())
        conv_u = _cuda(conv_u.transpose(0, 1).contiguous())
        selector_index = _cuda(selector_index.contiguous())
        ent_selector_index = _cuda(ent_selector_index.contiguous())
        conv_ent_mask = _cuda(conv_ent_mask.contiguous())
        ptr_index = _cuda(ptr_index.contiguous())
        conv_arr = _cuda(conv_arr.transpose(0, 1).contiguous())
        sketch_response = _cuda(sketch_response.contiguous())
        context_word_lengths = _cuda(context_word_lengths.contiguous())
        conv_word_lengths = _cuda(conv_word_lengths.contiguous())

        if len(list(kb_arr.size())) > 1:
            kb_arr = _cuda(kb_arr.transpose(0, 1).contiguous())
        item_info['label_arr'] = []

        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        # additional plain information
        data_info['context_arr_lengths'] = context_arr_lengths
        data_info['response_lengths'] = response_lengths
        data_info['conv_arr_lengths'] = conv_arr_lengths
        data_info['kb_arr_lengths'] = kb_arr_lengths
        data_info['conv_u_lengths'] = conv_u_lengths
        return data_info


def get_seq(pairs, lang, batch_size, type):
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []

    for pair in pairs:

        # if type and pair['domain'] == 'weather' and random.random() < 0.99:
        #     pass
        # else:
        for k in pair.keys():
            data_info[k].append(pair[k])

        if (type):
            lang.index_words(pair['context_arr'])
            lang.index_words(pair['response'], trg=True)
            lang.index_words(pair['sketch_response'], trg=True)

    dataset = Dataset(data_info, lang.word2index, lang.word2index, lang)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=dataset.collate_fn)
    return data_loader
