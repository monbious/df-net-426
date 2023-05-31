import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast

from utils.utils_general import *


def read_langs(file_name, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr = [], [], [], []
    max_resp_len = 0
    kb_source = []
    conv_arr_plain = []
    max_seq_len = 0

    conv_u, conv_ent_mask = [], []
    context_word_lengths, conv_word_lengths = [], []

    with open('data/MULTIWOZ2.1/global_entities.json') as f:
        global_entity = json.load(f)
        global_entity_keys = list(global_entity.keys())

    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:
                if line[-1] == line[0] == "#":
                    line = line.replace("#", "")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, gold_ent = line.split('\t')
                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)

                    sket_u_plain, _ = generate_template(global_entity, u, gold_ent, kb_arr, task_type)
                    gen_u, word_lens = generate_memory(u, "$u", str(nid), task_type, kb_arr, sket_u_plain)
                    context_arr += gen_u
                    conv_arr += gen_u
                    context_word_lengths += word_lens
                    conv_word_lengths += word_lens

                    conv_ent_mask += [1 if w in gold_ent else 2 for w in u.split()]

                    sket_u, _ = generate_memory(sket_u_plain, "$u", str(nid), task_type, kb_arr, sket_u_plain)
                    conv_u += sket_u

                    ent_idx_restaurant, ent_idx_attraction, ent_idx_hotel = [], [], []
                    if task_type == "restaurant":
                        ent_idx_restaurant = gold_ent
                    elif task_type == "attraction":
                        ent_idx_attraction = gold_ent
                    elif task_type == "hotel":
                        ent_idx_hotel = gold_ent
                    ent_index = list(set(ent_idx_restaurant + ent_idx_attraction + ent_idx_hotel))

                    # Get local pointer position for each word in system response
                    ptr_index = []
                    for key in r.split():
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in ent_index)]
                        if (index):
                            index = max(index)
                        else:
                            index = len(context_arr)
                        ptr_index.append(index)

                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    ent_history = [item for item in ent_index if item in r.split()]
                    selector_index = [1 if (word_arr[0] in r.split() and word_arr[0] not in ent_history) else 0
                                      for word_arr in context_arr] + [0]

                    ent_selector_index = [1 if (word_arr[0] in ent_history) else 0
                                          for word_arr in context_arr] + [1]

                    sketch_response, gold_sketch = generate_template(global_entity, r, gold_ent, kb_arr, task_type)

                    data_detail = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        'response': r,
                        'sketch_response': sketch_response,
                        'conv_u': list(conv_u),
                        'conv_ent_mask': list(conv_ent_mask),
                        'gold_sketch': gold_sketch,
                        'ptr_index': ptr_index + [len(context_arr)],
                        'selector_index': selector_index,
                        'ent_selector_index': ent_selector_index,
                        'ent_index': ent_index,
                        'ent_idx_restaurant': list(set(ent_idx_restaurant)),
                        'ent_idx_attraction': list(set(ent_idx_attraction)),
                        'ent_idx_hotel': list(set(ent_idx_hotel)),
                        'conv_arr': list(conv_arr),
                        'kb_arr': list(kb_arr),
                        'context_word_lengths': list(context_word_lengths),
                        'conv_word_lengths': list(conv_word_lengths),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type}

                    data.append(data_detail)

                    gen_r, word_lens = generate_memory(r, "$s", str(nid), task_type, kb_arr, sketch_response)
                    context_arr += gen_r
                    conv_arr += gen_r
                    context_word_lengths += word_lens
                    conv_word_lengths += word_lens

                    conv_ent_mask += [1 if w in ent_index else 2 for w in r.split()]

                    sket_r, _ = generate_memory(sketch_response, "$s", str(nid), task_type, kb_arr, sketch_response)
                    conv_u += sket_r

                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    r = line.strip()
                    kb_source.append(r.split(' '))

                    kb_info, word_lens = generate_memory(r, "", str(nid), task_type, kb_arr)
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
                    context_word_lengths = word_lens + context_word_lengths
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr = [], [], []
                kb_source = []
                conv_arr_plain = []
                conv_u = []
                conv_ent_mask = []
                context_word_lengths, conv_word_lengths = [], []
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def generate_template(global_entity, sentence, sent_ent, kb_arr, domain):
    """
    Based on the system response and the provided entity table, the output is the sketch response. 
    """
    sketch_response = []
    gold_sketch = []
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                for kb_item in kb_arr:
                    if word == kb_item[0]:
                        ent_type = kb_item[1]
                        break
                assert ent_type != None
                sketch_response.append('@' + ent_type)
                gold_sketch.append('@' + ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response, gold_sketch

def get_ent_type(word, kb_arr):
    ent_type = ''
    for kb_item in kb_arr:
        if word == kb_item[0]:
            ent_type = kb_item[1]
            break
    assert ent_type != ''
    return '@' + ent_type

def generate_memory(sent, speaker, time, task_type, kb_arr, sket_sent=None):
    sent_new = []
    sent_token = sent.split()
    word_lengths = []
    if speaker == "$u" or speaker == "$s":
        sket_sents = sket_sent.split()
        for idx, word in enumerate(sent_token):
            word_len = MEM_TOKEN_SIZE
            if '@' not in sket_sents[idx]:
                ent_format = 'PAD'
                word_len = 4
            else:
                ent_format = sket_sents[idx]
                word_len = 5
            temp = [word, speaker, 'turn' + str(time), 'word' + str(idx), ent_format] + ["PAD"] * (MEM_TOKEN_SIZE - 5)
            sent_new.append(temp)
            word_lengths.append(word_len)
    else:
        ent_format = '@' + sent_token[-2]
        sent_token = sent_token[::-1] + [ent_format] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token) - 1)
        sent_new.append(sent_token)
        word_lengths.append(len(sent_token) + 1)
    return sent_new, word_lengths


def prepare_data_seq(batch_size=100):
    file_train = 'data/MULTIWOZ2.1/train.txt'
    file_dev = 'data/MULTIWOZ2.1/dev.txt'
    file_test = 'data/MULTIWOZ2.1/test.txt'

    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True)
    dev = get_seq(pair_dev, lang, batch_size, False)
    test = get_seq(pair_test, lang, batch_size, False)

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, [], lang, max_resp_len


def get_data_seq(file_name, lang, max_len, batch_size=1):
    pair, _ = read_langs(file_name, max_line=None)
    d = get_seq(pair, lang, batch_size, False)
    return d
