from torch.optim import lr_scheduler
import json
import random

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch import optim
from torch.optim import lr_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Config

from models.modules import *
from utils.masked_cross_entropy import *
from utils.measures import moses_multi_bleu


class DFNet(nn.Module):
    def __init__(self, hidden_size, lang, max_resp_len, path, lr, n_layers, dropout, domains=None, max_seq_len=500,
                 tf_num_layers=4, tf_num_heads=8, tokenizer=None):
        super(DFNet, self).__init__()
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_resp_len = max_resp_len
        self.decoder_hop = n_layers
        self.softmax = nn.Softmax(dim=0)
        self.domains = domains

        self.gpt2_config = GPT2Config.from_pretrained('gpt2')
        self.gpt2_config.n_embd = 128
        self.gpt2_config.n_layer = 4
        self.gpt2_config.n_head = 8

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th')
                self.gptModel = AutoModelForCausalLM.from_pretrained(str(path))
                self.tokenizer = AutoTokenizer.from_pretrained(str(path))
                self.extKnow = torch.load(str(path) + '/enc_kb.th')
                self.decoder = torch.load(str(path) + '/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
                self.gptModel = AutoModelForCausalLM.from_pretrained(str(path))
                self.tokenizer = AutoTokenizer.from_pretrained(str(path))
                self.extKnow = torch.load(str(path) + '/enc_kb.th', lambda storage, loc: storage)
                self.decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)
        else:
            self.encoder = ContextEncoder(lang.n_words, hidden_size, dropout, domains)
            self.gptModel = GPT2LMHeadModel(self.gpt2_config)
            self.tokenizer = tokenizer
            self.gptModel.resize_token_embeddings(len(self.tokenizer))
            self.extKnow = ExternalKnowledge(lang.n_words, hidden_size, n_layers, dropout)
            self.decoder = LocalMemoryDecoder(self.encoder.embedding, lang, hidden_size, self.decoder_hop,
                                              dropout, domains=domains)

        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.gptModel_optimizer = optim.Adam(self.gptModel.parameters(), lr=lr)
        self.extKnow_optimizer = optim.Adam(self.extKnow.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)
        self.gpt_scheduler = lr_scheduler.ReduceLROnPlateau(self.gptModel_optimizer, mode='max', factor=0.5, patience=1,
                                                            min_lr=0.00001, verbose=True)
        self.criterion_bce = nn.BCELoss()
        self.criterion_label = nn.BCELoss()
        self.cross_entr_loss = nn.CrossEntropyLoss()
        self.reset()
        if USE_CUDA:
            self.encoder.cuda()
            self.extKnow.cuda()
            self.decoder.cuda()
            self.gptModel.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_g = self.loss_g / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_l = self.loss_l / self.print_every
        print_loss_t = self.loss_t / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LE:{:.2f},LG:{:.2f},LP:{:.2f},LT:{:.2f}' \
            .format(print_loss_avg, print_loss_g, print_loss_v, print_loss_l, print_loss_t)

    def save_model(self, dec_type):
        if args['dataset'] == 'kvr':
            name_data = "KVR/"
        elif args['dataset'] == 'woz':
            name_data = "WOZ/"
        layer_info = str(self.n_layers)
        directory = 'save/DF-Net-' + args["addName"] + name_data + 'HDD' + str(
            self.hidden_size) + 'BSZ' + str(args['batch']) + 'DR' + str(self.dropout) + 'L' + layer_info + 'lr' + str(
            self.lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        args['path'] = directory
        torch.save(self.encoder, directory + '/enc.th')
        self.gptModel.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        torch.save(self.extKnow, directory + '/enc_kb.th')
        torch.save(self.decoder, directory + '/dec.th')

    def reset(self):
        self.loss, self.print_every, self.loss_g, self.loss_v, self.loss_l, self.loss_t = 0, 1, 0, 0, 0, 0

    def _cuda(self, x):
        if USE_CUDA:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)

    def train_batch(self, data, clip, reset=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.gptModel_optimizer.zero_grad()
        self.extKnow_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Encode and Decode
        use_teacher_forcing = random.random() < args['teacher_forcing_ratio']
        max_target_length = max(data['response_lengths'])
        all_decoder_outputs_vocab, all_decoder_outputs_ptr, _, _, global_pointer, \
        label_e, label_d, label_mix_e, label_mix_d, gpt_loss = self.encode_and_decode(
            data, max_target_length, use_teacher_forcing, False)

        # Loss calculation and backpropagation
        domains = []
        for domain in data['domain']:
            domains.append(self.domains[domain])
        # print(global_pointer)
        loss_g = self.criterion_bce(global_pointer, data['selector_index'])
        # print(all_decoder_outputs_vocab.transpose(0, 1).shape)
        # print(transformer_output.shape)
        loss_v = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),
            data['sketch_response'].contiguous(),
            data['response_lengths'])
        loss_l = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(),
            data['ptr_index'].contiguous(),
            data['response_lengths'])

        # print('loss_l: ', loss_l)
        # loss_t = self.cross_entr_loss(transformer_output.contiguous().view(-1, self.output_size),
        #                               data['sketch_response_tf'][:, 1:].contiguous().view(-1))
        # print('loss_t: ', loss_t)
        loss_t = gpt_loss
        loss = loss_g + loss_v + loss_l + loss_t

        golden_labels = torch.zeros_like(label_e).scatter_(1, data['label_arr'], 1)
        loss += self.criterion_label(label_e, golden_labels)
        loss += self.criterion_label(label_d, golden_labels)

        domains = self._cuda(torch.Tensor(domains)).long().unsqueeze(-1)
        loss += masked_cross_entropy(label_mix_e, domains.expand(len(domains), label_mix_e.size(1)).contiguous(),
                                     data['conv_arr_lengths'])
        loss += masked_cross_entropy(label_mix_d, domains.expand(len(domains), label_mix_d.size(1)).contiguous(),
                                     data['response_lengths'])
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        ec = torch.nn.utils.clip_grad_norm_(self.extKnow.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        tf = torch.nn.utils.clip_grad_norm_(self.gptModel.parameters(), clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.gptModel_optimizer.step()
        self.extKnow_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_g += loss_g.item()
        self.loss_v += loss_v.item()
        self.loss_l += loss_l.item()
        self.loss_t += loss_t.item()

    def get_sentence_bleu(self, pred_text, refer_text):
        predicted_text = self.gpt2_tokenizer.tokenize(pred_text)
        reference_text = [self.gpt2_tokenizer.tokenize(refer_text)]
        # print(reference_text)
        bleu_score = sentence_bleu(reference_text, predicted_text, smoothing_function=SmoothingFunction.method1)
        return bleu_score

    def process_extract_res(self, data, max_target_length):
        g_inputs, g_mask = self._cuda(data['g_input_ids_padded']), self._cuda(data['g_attention_masks'])
        output_ids = self.gptModel.generate(input_ids=g_inputs, pad_token_id=self.tokenizer.eos_token_id,
                                            attention_mask=g_mask,
                                            max_new_tokens=max_target_length, do_sample=True, num_beams=5,
                                            temperature=1.0)
        predicted_texts = [self.tokenizer.decode(output_id, skip_special_tokens=True) for output_id in
                           output_ids]
        bat_token_arr = []
        for i, pred_text in enumerate(predicted_texts):
            # print('pred_text: ', pred_text)
            pred_resp = pred_text.split('[SEP]')[-1].strip()
            # if not self.gptModel.training:
            #     print('='*50)
            #     print('  pred_resp: ', pred_resp)
            #     print('target_resp: ', data['answer_texts'][i])
            # bleu = self.get_sentence_bleu(pred_resp, data['answer_texts'][i])
            sen_arr = []
            for token in self.tokenizer.tokenize(pred_resp):
                token = token.replace('Ġ', '')
                try:
                    lang_token_id = self.lang.word2index[token]
                except Exception:
                    lang_token_id = UNK_token
                sen_arr.append(lang_token_id)
            bat_token_arr.append(sen_arr)
        max_len = max(len(sen) for sen in bat_token_arr)
        bat_token_arr = [ids + [PAD_token] * (max_len - len(ids)) for ids in bat_token_arr]
        tmp_resp = self._cuda(torch.LongTensor(bat_token_arr))
        return tmp_resp

    def encode_and_decode(self, data, max_target_length, use_teacher_forcing, get_decoded_words,
                          global_entity_type=None):
        # Build unknown mask for memory
        if args['unk_mask'] and self.decoder.training:
            story_size = data['context_arr'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            rand_mask[:, :, 0] = rand_mask[:, :, 0] * bi_mask
            conv_rand_mask = np.ones(data['conv_arr'].size())
            for bi in range(story_size[0]):
                start, end = data['kb_arr_lengths'][bi], data['kb_arr_lengths'][bi] + data['conv_arr_lengths'][bi]
                conv_rand_mask[:end - start, bi, :] = rand_mask[bi, start:end, :]
            rand_mask = self._cuda(rand_mask)
            conv_rand_mask = self._cuda(conv_rand_mask)
            conv_story = data['conv_arr'] * conv_rand_mask.long()
            story = data['context_arr'] * rand_mask.long()
        else:
            story, conv_story = data['context_arr'], data['conv_arr']

        dh_outputs, dh_hidden, label_e, label_mix_e = self.encoder(conv_story, data['conv_arr_lengths'])

        if self.gptModel.training:
            conv_inputs, mask = self._cuda(data['input_ids_padded']), self._cuda(data['attention_masks'])
            gpt_loss = self.gptModel(conv_inputs, attention_mask=mask, labels=conv_inputs)[0]

            self.tokenizer.padding_side = 'left'
            tmp_resp = self.process_extract_res(data, max_target_length)
            self.tokenizer.padding_side = 'right'
        else:
            gpt_loss = None
            tmp_resp = self.process_extract_res(data, max_target_length)

        global_pointer, kb_readout = self.extKnow.load_memory(story, data['kb_arr_lengths'], data['conv_arr_lengths'],
                                                              dh_hidden, dh_outputs, data['domain'], tmp_resp)
        encoded_hidden = torch.cat((dh_hidden, kb_readout), dim=1)

        # Get the words that can be copy from the memory
        batch_size = len(data['context_arr_lengths'])
        self.copy_list = []
        for elm in data['context_arr_plain']:
            elm_temp = [word_arr[0] for word_arr in elm]
            self.copy_list.append(elm_temp)
        # print('======>', data['context_arr_plain'])

        outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse, label_d, label_mix_d = self.decoder.forward(
            self.extKnow,
            story.size(),
            data['context_arr_lengths'],
            self.copy_list,
            encoded_hidden,
            data['sketch_response'],
            max_target_length,
            batch_size,
            use_teacher_forcing,
            get_decoded_words,
            global_pointer,
            H=dh_outputs,
            global_entity_type=global_entity_type,
            domains=data['label_arr'],
            kb_readout=kb_readout,
            tmp_resp=tmp_resp)

        return outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse, global_pointer, \
               label_e, label_d, label_mix_e, label_mix_d, gpt_loss

    def evaluate(self, dev, matric_best, output=False, early_stop=None):
        print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.gptModel.train(False)
        self.extKnow.train(False)
        self.decoder.train(False)
        self.tokenizer.padding_side = 'left'

        ref, hyp = [], []
        ids = []
        acc, total = 0, 0
        if args['dataset'] == 'kvr':
            F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
            F1_count, F1_cal_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
            TP_all, FP_all, FN_all = 0, 0, 0

            TP_sche, FP_sche, FN_sche = 0, 0, 0
            TP_wea, FP_wea, FN_wea = 0, 0, 0
            TP_nav, FP_nav, FN_nav = 0, 0, 0
        elif args['dataset'] == 'woz':
            F1_pred, F1_police_pred, F1_restaurant_pred, F1_hospital_pred, F1_attraction_pred, F1_hotel_pred = 0, 0, 0, 0, 0, 0
            F1_count, F1_police_count, F1_restaurant_count, F1_hospital_count, F1_attraction_count, F1_hotel_count = 0, 0, 0, 0, 0, 0
            TP_all, FP_all, FN_all = 0, 0, 0

            TP_restaurant, FP_restaurant, FN_restaurant = 0, 0, 0
            TP_attraction, FP_attraction, FN_attraction = 0, 0, 0
            TP_hotel, FP_hotel, FN_hotel = 0, 0, 0

        pbar = tqdm(enumerate(dev), total=len(dev))

        if args['dataset'] == 'kvr':
            entity_path = 'data/KVR/kvret_entities.json'
        elif args['dataset'] == 'woz':
            entity_path = 'data/MULTIWOZ2.1/global_entities.json'
        else:
            print('dataset args error')
            exit(1)

        with open(entity_path) as f:
            global_entity = json.load(f)
            global_entity_type = {}
            global_entity_list = []
            for key in global_entity.keys():
                if key != 'poi':
                    entity_arr = [item.lower().replace(' ', '_') for item in global_entity[key]]
                    global_entity_list += entity_arr
                    for entity in entity_arr:
                        global_entity_type[entity] = key
                else:
                    for item in global_entity['poi']:
                        entity_arr = [item[k].lower().replace(' ', '_') for k in item.keys()]
                        global_entity_list += entity_arr
                        for key in item:
                            global_entity_type[item[key].lower().replace(' ', '_')] = key
            global_entity_list = list(set(global_entity_list))

        for j, data_dev in pbar:
            ids.extend(data_dev['id'])
            # Encode and Decode
            _, _, decoded_fine, decoded_coarse, global_pointer, _, _, _, _, _ = self.encode_and_decode(data_dev,
                                                                                                       self.max_resp_len,
                                                                                                       False,
                                                                                                       True,
                                                                                                       global_entity_type)
            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS':
                        break
                    else:
                        st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS':
                        break
                    else:
                        st_c += e + ' '
                pred_sent = st.lstrip().rstrip()
                pred_sent_coarse = st_c.lstrip().rstrip()
                gold_sent = data_dev['response_plain'][bi].lstrip().rstrip()
                ref.append(gold_sent)
                hyp.append(pred_sent)

                if args['dataset'] == 'kvr':
                    # compute F1 SCORE
                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_index'][bi],
                                                                                         pred_sent.split(),
                                                                                         global_entity_list,
                                                                                         data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    TP_all += single_tp
                    FP_all += single_fp
                    FN_all += single_fn

                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_idx_cal'][bi],
                                                                                         pred_sent.split(),
                                                                                         global_entity_list,
                                                                                         data_dev['kb_arr_plain'][bi])
                    F1_cal_pred += single_f1
                    F1_cal_count += count
                    TP_sche += single_tp
                    FP_sche += single_fp
                    FN_sche += single_fn

                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_idx_nav'][bi],
                                                                                         pred_sent.split(),
                                                                                         global_entity_list,
                                                                                         data_dev['kb_arr_plain'][bi])
                    F1_nav_pred += single_f1
                    F1_nav_count += count
                    TP_nav += single_tp
                    FP_nav += single_fp
                    FN_nav += single_fn

                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_idx_wet'][bi],
                                                                                         pred_sent.split(),
                                                                                         global_entity_list,
                                                                                         data_dev['kb_arr_plain'][bi])
                    F1_wet_pred += single_f1
                    F1_wet_count += count
                    TP_wea += single_tp
                    FP_wea += single_fp
                    FN_wea += single_fn

                elif args['dataset'] == 'woz':
                    # coimpute F1 SCORE
                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_index'][bi],
                                                                                         pred_sent.split(),
                                                                                         global_entity_list,
                                                                                         data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    TP_all += single_tp
                    FP_all += single_fp
                    FN_all += single_fn

                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(
                        data_dev['ent_idx_restaurant'][bi],
                        pred_sent.split(),
                        global_entity_list,
                        data_dev['kb_arr_plain'][bi])
                    F1_restaurant_pred += single_f1
                    F1_restaurant_count += count
                    TP_restaurant += single_tp
                    FP_restaurant += single_fp
                    FN_restaurant += single_fn

                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(
                        data_dev['ent_idx_attraction'][bi],
                        pred_sent.split(),
                        global_entity_list,
                        data_dev['kb_arr_plain'][bi])
                    F1_attraction_pred += single_f1
                    F1_attraction_count += count
                    TP_attraction += single_tp
                    FP_attraction += single_fp
                    FN_attraction += single_fn

                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_idx_hotel'][bi],
                                                                                         pred_sent.split(),
                                                                                         global_entity_list,
                                                                                         data_dev['kb_arr_plain'][bi])
                    F1_hotel_pred += single_f1
                    F1_hotel_count += count
                    TP_hotel += single_tp
                    FP_hotel += single_fp
                    FN_hotel += single_fn

                # compute Per-response Accuracy Score
                total += 1
                if (gold_sent == pred_sent):
                    acc += 1

                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent)

        # Set back to training mode
        self.encoder.train(True)
        self.gptModel.train(True)
        self.extKnow.train(True)
        self.decoder.train(True)
        self.tokenizer.padding_side = 'right'

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        acc_score = acc / float(total)
        print("ACC SCORE:\t" + str(acc_score))

        if args['dataset'] == 'kvr':
            print("BLEU SCORE:\t" + str(bleu_score))
            print("F1-macro SCORE:\t{}".format(F1_pred / float(F1_count)))
            print("F1-macro-schedule SCORE:\t{}".format(F1_cal_pred / float(F1_cal_count)))
            print("F1-macro-weather SCORE:\t{}".format(F1_wet_pred / float(F1_wet_count)))
            print("F1-macro-navigate SCORE:\t{}".format(F1_nav_pred / float(F1_nav_count)))

            P_score = TP_all / float(TP_all + FP_all) if (TP_all + FP_all) != 0 else 0
            R_score = TP_all / float(TP_all + FN_all) if (TP_all + FN_all) != 0 else 0
            P_nav_score = TP_nav / float(TP_nav + FP_nav) if (TP_nav + FP_nav) != 0 else 0
            P_sche_score = TP_sche / float(TP_sche + FP_sche) if (TP_sche + FP_sche) != 0 else 0
            P_wea_score = TP_wea / float(TP_wea + FP_wea) if (TP_wea + FP_wea) != 0 else 0
            R_nav_score = TP_nav / float(TP_nav + FN_nav) if (TP_nav + FN_nav) != 0 else 0
            R_sche_score = TP_sche / float(TP_sche + FN_sche) if (TP_sche + FN_sche) != 0 else 0
            R_wea_score = TP_wea / float(TP_wea + FN_wea) if (TP_wea + FN_wea) != 0 else 0

            F1_score = self.compute_F1(P_score, R_score)
            print("F1-micro SCORE:\t{}".format(F1_score))
            print("F1-micro-schedule SCORE:\t{}".format(self.compute_F1(P_sche_score, R_sche_score)))
            print("F1-micro-weather SCORE:\t{}".format(self.compute_F1(P_wea_score, R_wea_score)))
            print("F1-micro-navigate SCORE:\t{}".format(self.compute_F1(P_nav_score, R_nav_score)))
        elif args['dataset'] == 'woz':
            print("BLEU SCORE:\t" + str(bleu_score))
            print("F1-macro SCORE:\t{}".format(F1_pred / float(F1_count)))
            print("F1-macro-restaurant SCORE:\t{}".format(F1_restaurant_pred / float(F1_restaurant_count)))
            print("F1-macro-attraction SCORE:\t{}".format(F1_attraction_pred / float(F1_attraction_count)))
            print("F1-macro-hotel SCORE:\t{}".format(F1_hotel_pred / float(F1_hotel_count)))

            P_score = TP_all / float(TP_all + FP_all) if (TP_all + FP_all) != 0 else 0
            R_score = TP_all / float(TP_all + FN_all) if (TP_all + FN_all) != 0 else 0
            P_restaurant_score = TP_restaurant / float(TP_restaurant + FP_restaurant) if (
                                                                                                 TP_restaurant + FP_restaurant) != 0 else 0
            P_attraction_score = TP_attraction / float(TP_attraction + FP_attraction) if (
                                                                                                 TP_attraction + FP_attraction) != 0 else 0
            P_hotel_score = TP_hotel / float(TP_hotel + FP_hotel) if (TP_hotel + FP_hotel) != 0 else 0

            R_restaurant_score = TP_restaurant / float(TP_restaurant + FN_restaurant) if (
                                                                                                 TP_restaurant + FN_restaurant) != 0 else 0
            R_attraction_score = TP_attraction / float(TP_attraction + FN_attraction) if (
                                                                                                 TP_attraction + FN_attraction) != 0 else 0
            R_hotel_score = TP_hotel / float(TP_hotel + FN_hotel) if (TP_hotel + FN_hotel) != 0 else 0

            F1_score = self.compute_F1(P_score, R_score)
            print("F1-micro SCORE:\t{}".format(F1_score))
            print("F1-micro-restaurant SCORE:\t{}".format(self.compute_F1(P_restaurant_score, R_restaurant_score)))
            print("F1-micro-attraction SCORE:\t{}".format(self.compute_F1(P_attraction_score, R_attraction_score)))
            print("F1-micro-hotel SCORE:\t{}".format(self.compute_F1(P_hotel_score, R_hotel_score)))

        if output:
            print('Test Finish!')
            with open(args['output'], 'w+') as f:
                if args['dataset'] == 'kvr':
                    print("ACC SCORE:\t" + str(acc_score), file=f)
                    print("BLEU SCORE:\t" + str(bleu_score), file=f)
                    print("F1-macro SCORE:\t{}".format(F1_pred / float(F1_count)), file=f)
                    print("F1-micro SCORE:\t{}".format(self.compute_F1(P_score, R_score)), file=f)
                    print("F1-macro-sche SCORE:\t{}".format(F1_cal_pred / float(F1_cal_count)), file=f)
                    print("F1-macro-wea SCORE:\t{}".format(F1_wet_pred / float(F1_wet_count)), file=f)
                    print("F1-macro-nav SCORE:\t{}".format(F1_nav_pred / float(F1_nav_count)), file=f)
                    print("F1-micro-sche SCORE:\t{}".format(self.compute_F1(P_sche_score, R_sche_score)), file=f)
                    print("F1-micro-wea SCORE:\t{}".format(self.compute_F1(P_wea_score, R_wea_score)), file=f)
                    print("F1-micro-nav SCORE:\t{}".format(self.compute_F1(P_nav_score, R_nav_score)), file=f)
                elif args['dataset'] == 'woz':
                    print("ACC SCORE:\t" + str(acc_score), file=f)
                    print("BLEU SCORE:\t" + str(bleu_score), file=f)
                    print("F1-macro SCORE:\t{}".format(F1_pred / float(F1_count)), file=f)
                    print("F1-micro SCORE:\t{}".format(self.compute_F1(P_score, R_score)), file=f)
                    print("F1-macro-restaurant SCORE:\t{}".format(F1_restaurant_pred / float(F1_restaurant_count)),
                          file=f)
                    print("F1-macro-attraction SCORE:\t{}".format(F1_attraction_pred / float(F1_attraction_count)),
                          file=f)
                    print("F1-macro-hotel SCORE:\t{}".format(F1_hotel_pred / float(F1_hotel_count)), file=f)
                    print("F1-micro-restaurant SCORE:\t{}".format(
                        self.compute_F1(P_restaurant_score, R_restaurant_score)),
                        file=f)
                    print("F1-micro-attraction SCORE:\t{}".format(
                        self.compute_F1(P_attraction_score, R_attraction_score)),
                        file=f)
                    print("F1-micro-hotel SCORE:\t{}".format(self.compute_F1(P_hotel_score, R_hotel_score)), file=f)

        if (early_stop == 'BLEU'):
            if (bleu_score >= matric_best):
                self.save_model('BLEU-' + str(bleu_score) + 'F1-' + str(F1_score))
                print("MODEL SAVED")
            return bleu_score
        elif (early_stop == 'ENTF1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")
            return F1_score
        else:
            if (acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(acc_score))
                print("MODEL SAVED")
            return acc_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return TP, FP, FN, F1, count

    def compute_F1(self, precision, recall):
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        return F1

    def print_examples(self, batch_idx, data, pred_sent, pred_sent_coarse, gold_sent):
        kb_len = len(data['context_arr_plain'][batch_idx]) - data['conv_arr_lengths'][batch_idx] - 1
        print("{}: ID{} id{} ".format(data['domain'][batch_idx], data['ID'][batch_idx], data['id'][batch_idx]))
        for i in range(kb_len):
            kb_temp = [w for w in data['context_arr_plain'][batch_idx][i] if w != 'PAD']
            kb_temp = kb_temp[::-1]
            if 'poi' not in kb_temp:
                print(kb_temp)
        flag_uttr, uttr = '$u', []
        for word_idx, word_arr in enumerate(data['context_arr_plain'][batch_idx][kb_len:]):
            if word_arr[1] == flag_uttr:
                uttr.append(word_arr[0])
            else:
                print(flag_uttr, ': ', " ".join(uttr))
                flag_uttr = word_arr[1]
                uttr = [word_arr[0]]
        print('Sketch System Response : ', pred_sent_coarse)
        print('Final System Response : ', pred_sent)
        print('Gold System Response : ', gold_sent)
        print('\n')
