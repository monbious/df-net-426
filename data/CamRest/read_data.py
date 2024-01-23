import os
import json
from nltk import wordpunct_tokenize as tokenizer
import argparse

def cleaner(token_array):
    new_token_array = []
    for idx, token in enumerate(token_array):
        temp = token
        if token==".." or token=="." or token=="...": continue
        if token==":),": continue
        new_token_array.append(temp)
    return new_token_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--json', dest='json',
                        default='CamRest676_dev.json',
                        help='process json file')
    parser.add_argument('--output_file', dest='output_file',
                        default='dev_1.txt',
                        help='process json file')
    args = parser.parse_args()

    with open(args.json) as f:
        dialogues = json.load(f)

    column_names = ["name", "address", "area", "food", "id", "location", "phone", "postcode", "pricerange", "type"]

    with open(args.output_file, 'a', encoding='utf-8') as f:
        for d in dialogues:
            print("#restaurant#", file=f)
            #kb
            entity_set = []
            for kb in d['scenario']['kb']['items']:
                for c in column_names:
                    entity_set.append(str(kb[c]).lower())
                    if c != "name":
                        print("0 "+str(kb['name']).lower()+" "+c+" "+str(kb[c]).lower()+" name "+c, file=f)
            entity_set = list(set(entity_set))

            #dialog
            if (len(d['dialogue'])%2 != 0):
                d['dialogue'].pop()

            j = 1
            for i in range(0, len(d['dialogue']), 2):
                user = " ".join(cleaner(tokenizer(str(d['dialogue'][i]['data']['utterance']).lower())))
                bot = " ".join(cleaner(tokenizer(str(d['dialogue'][i+1]['data']['utterance']).lower())))
                gold_entity = []
                for key in bot.split(' '):
                    if key in entity_set:
                        gold_entity.append(key)
                gold_entity = list(set(gold_entity))
                if user!="" and bot!="":
                    print(str(j)+" "+user+'\t'+bot+'\t'+str(gold_entity), file=f)
                    j+=1
            print("", file=f)
