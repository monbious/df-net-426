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
                    gen_u = generate_memory(u, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u
                    conv_arr_plain.append(u)

                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)

                    for i, ent in enumerate(gold_ent):
                        if ent in u:
                            ref = list(set([t for tup in kb_source if (ent in tup)
                                            for t in tup if t not in global_entity_keys and t != ent]))
                            context_arr.append(
                                [ent, "$u", 'turn' + str(nid), 'ent' + str(i)] + ["PAD"] * (MEM_TOKEN_SIZE - 4))
                            conv_arr.append(
                                [ent, "$u", 'turn' + str(nid), 'ent' + str(i)] + ["PAD"] * (MEM_TOKEN_SIZE - 4))
                            for refer in ref:
                                context_arr.append(
                                    [refer, "$u", 'turn' + str(nid), 'ref' + str(i)] + ["PAD"] * (MEM_TOKEN_SIZE - 4))
                                conv_arr.append(
                                    [refer, "$u", 'turn' + str(nid), 'ref' + str(i)] + ["PAD"] * (MEM_TOKEN_SIZE - 4))

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
                    selector_index = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in
                                      context_arr] + [1]

                    sketch_response, gold_sketch = generate_template(global_entity, r, gold_ent, kb_arr, task_type)
                    sketch_response_tf = 'SOS ' + sketch_response
                    conv_plain = ' '.join(conv_arr_plain)
                    data_detail = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        'response': r,
                        'sketch_response': sketch_response,
                        'sketch_response_tf': sketch_response_tf,
                        'gold_sketch': gold_sketch,
                        'ptr_index': ptr_index + [len(context_arr)],
                        'selector_index': selector_index,
                        'ent_index': ent_index,
                        'ent_idx_restaurant': list(set(ent_idx_restaurant)),
                        'ent_idx_attraction': list(set(ent_idx_attraction)),
                        'ent_idx_hotel': list(set(ent_idx_hotel)),
                        'conv_arr': list(conv_arr),
                        'conv_arr_plain': conv_plain,
                        'kb_arr': list(kb_arr),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type}

                    data.append(data_detail)
                    conv_arr_plain.append(r)

                    gen_r = generate_memory(r, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r

                    if max_seq_len < len(conv_plain.split()):
                        max_seq_len = len(conv_plain.split())
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    r = line.strip()
                    kb_source.append(r.split(' '))

                    kb_info = generate_memory(r, "", str(nid))
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr = [], [], []
                kb_source = []
                conv_arr_plain = []
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len, max_seq_len


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


def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn' + str(time), 'word' + str(idx)] + ["PAD"] * (MEM_TOKEN_SIZE - 4)
            sent_new.append(temp)
    else:
        sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new


def prepare_data_seq(batch_size=100):
    file_train = 'data/MULTIWOZ2.1/train.txt'
    file_dev = 'data/MULTIWOZ2.1/dev.txt'
    file_test = 'data/MULTIWOZ2.1/test.txt'

    pair_train, train_max_len, trn_seq_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len, dev_seq_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len, test_seq_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1
    max_seq_len = max(trn_seq_len, dev_seq_len, test_seq_len)

    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True)
    dev = get_seq(pair_dev, lang, batch_size, False)
    test = get_seq(pair_test, lang, batch_size, False)

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("Max. length of input: %s " % max_seq_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, [], lang, max_resp_len, max_seq_len


def get_data_seq(file_name, lang, max_len, batch_size=1):
    pair, _ = read_langs(file_name, max_line=None)
    d = get_seq(pair, lang, batch_size, False)
    return d
