import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast
import random

from utils.utils_general import *


def read_langs(file_name, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr = [], [], [], []
    max_resp_len = 0
    kb_source = []
    kb_plains = []
    counter_set1, counter_set2 = set(), set()
    ent_history = []
    conv_u = []

    with open('data/KVR/kvret_entities.json') as f:
        global_entity = json.load(f)
        global_entity_keys = list(global_entity.keys()) + list(global_entity['poi'][0].keys())

    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:
                if '#' in line:
                    line = line.replace("#", "")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)

                if '\t' in line:
                    u, r, gold_ent = line.split('\t')
                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)

                    sket_u_plain, _ = generate_template(global_entity, u, gold_ent, kb_arr, task_type)
                    gen_u = generate_memory(u, "$u", str(nid), task_type, global_entity, sket_u_plain)
                    context_arr += gen_u
                    conv_arr += gen_u

                    sket_u = generate_memory(sket_u_plain, "$u", str(nid), task_type, global_entity, sket_u_plain)
                    conv_u += sket_u

                    ent_idx_cal, ent_idx_nav, ent_idx_wet = [], [], []
                    if task_type == "weather":
                        ent_idx_wet = gold_ent
                    elif task_type == "schedule":
                        ent_idx_cal = gold_ent
                    elif task_type == "navigate":
                        ent_idx_nav = gold_ent
                    ent_index = list(set(ent_idx_cal + ent_idx_nav + ent_idx_wet))
                    ent_history += ent_index

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
                    #  or word_arr[0] in r.split()
                    # ent_history = list(set(ent_history))
                    ent_history = [item for item in ent_index if item in r.split()]
                    selector_index = [1 if (word_arr[0] in ent_history) else 0
                                      for word_arr in context_arr] + [0]

                    sketch_response, gold_sketch = generate_template(global_entity, r, gold_ent, kb_arr, task_type)

                    kb_txt = ' '.join(kb_plains)
                    data_detail = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        'response': r,
                        'sketch_response': sketch_response,
                        'conv_u': list(conv_u),
                        'gold_sketch': gold_sketch,
                        'ptr_index': ptr_index + [len(context_arr)],
                        'selector_index': selector_index,
                        'ent_index': ent_index,
                        'ent_idx_cal': list(set(ent_idx_cal)),
                        'ent_idx_nav': list(set(ent_idx_nav)),
                        'ent_idx_wet': list(set(ent_idx_wet)),
                        'conv_arr': list(conv_arr),
                        'kb_arr': list(kb_arr),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type}

                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(nid), task_type, global_entity, sketch_response)
                    context_arr += gen_r
                    conv_arr += gen_r

                    sket_r = generate_memory(sketch_response, "$s", str(nid), task_type, global_entity, sketch_response)
                    conv_u += sket_r

                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    r = line.strip()
                    r_split = r.split(' ')
                    kb_source.append(r_split)

                    if len(r_split) < 5:
                        if r_split[0] not in counter_set1:
                            counter_set2.clear()
                            if len(kb_plains) > 0:
                                kb_plains.append("SEP")
                            kb_plains += r_split
                            counter_set1.add(r_split[0])
                            counter_set2.add(r_split[1])
                        else:
                            if r_split[1] not in counter_set2:
                                kb_plains += r_split[1:]
                                counter_set2.add(r_split[1])
                            else:
                                kb_plains += r_split[2:]

                    kb_info = generate_memory(r, "", str(nid), task_type, global_entity)
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info

            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr = [], [], []
                kb_source = []
                kb_plains = []
                counter_set1.clear()
                counter_set2.clear()
                ent_history = []
                conv_u = []
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
                if domain != 'weather':
                    for kb_item in kb_arr:
                        if word == kb_item[0]:
                            ent_type = kb_item[1]
                            break
                if ent_type == None:
                    for key in global_entity.keys():
                        if key != 'poi':
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                                ent_type = key
                                break
                        else:
                            poi_list = [d['poi'].lower() for d in global_entity['poi']]
                            if word in poi_list or word.replace('_', ' ') in poi_list:
                                ent_type = key
                                break
                # print(word)
                # print(ent_type)
                sketch_response.append('@' + ent_type)
                gold_sketch.append('@' + ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response, gold_sketch


def get_ent_type(word, global_entity):
    ent_type = ''
    for key in global_entity.keys():
        if key != 'poi':
            global_entity[key] = [x.lower() for x in global_entity[key]]
            if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                ent_type = key
                break
        else:
            poi_list = [d['poi'].lower() for d in global_entity['poi']]
            if word in poi_list or word.replace('_', ' ') in poi_list:
                ent_type = key
                break
    return '@' + ent_type


def generate_memory(sent, speaker, time, task_type, global_entity, sket_sent=None):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        sket_sents = sket_sent.split()
        for idx, word in enumerate(sent_token):
            if '@' not in sket_sents[idx]:
                ent_format = 'PAD'
            else:
                ent_format = sket_sents[idx]
            temp = [word, speaker, 'turn' + str(time), 'word' + str(idx), ent_format] + ["PAD"] * (MEM_TOKEN_SIZE - 5)
            sent_new.append(temp)
    else:
        ent_format = None
        if task_type != 'weather':
            ent_format = '@' + sent_token[-2]
        if ent_format is None:
            ent_format = get_ent_type(sent_token[-1], global_entity)
        sent_token = sent_token[::-1] + [ent_format] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token) - 1)
        sent_new.append(sent_token)
    return sent_new


def prepare_data_seq(batch_size=100):
    file_train = 'data/KVR/train.txt'
    file_dev = 'data/KVR/dev.txt'
    file_test = 'data/KVR/test.txt'

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
