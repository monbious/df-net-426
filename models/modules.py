import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.utils_general import _cuda
import math
import copy


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class RNN_Residual(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0., batch_first=True):
        super(RNN_Residual, self).__init__()
        for i in range(n_layers):
            if i == 0:
                setattr(self, 'forward_rnn_{}'.format(i), nn.GRU(input_dim, hidden_dim))
                setattr(self, 'backward_rnn_{}'.format(i), nn.GRU(input_dim, hidden_dim))
            else:
                setattr(self, 'forward_rnn_{}'.format(i), nn.GRU(hidden_dim, hidden_dim))
                setattr(self, 'backward_rnn_{}'.format(i), nn.GRU(hidden_dim, hidden_dim))
        self.n_layers = n_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first

    def run_rnn(self, rnn, embedded, input_lengths, batch_first=True, hx=None):
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=batch_first,
                                                     enforce_sorted=False)
        outputs, hidden = rnn(embedded, hx)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first)
        return outputs, hidden

    def flipByLength(self, input, lengths):
        output = _cuda(torch.zeros_like(input))
        for i, l in enumerate(lengths):
            output[i, :l, :] = torch.flip(input[i, :l, :], (0,))
        return output

    def forward(self, input, input_lengths, hx=None):
        input_forward = input
        input_backward = self.flipByLength(input, input_lengths)

        for i in range(self.n_layers):
            output_forward, hidden_forward = self.run_rnn(self.__getattr__('forward_rnn_{}'.format(i)), input_forward,
                                                          input_lengths, self.batch_first, hx)
            output_backward, hidden_backward = self.run_rnn(self.__getattr__('backward_rnn_{}'.format(i)),
                                                            input_backward, input_lengths, self.batch_first, hx)
            if i == 0:
                input_forward = F.dropout(output_forward, self.dropout, self.training)
                input_backward = F.dropout(output_backward, self.dropout, self.training)
            else:
                input_forward = F.dropout(output_forward + input_forward, self.dropout, self.training)
                input_backward = F.dropout(output_backward + input_backward, self.dropout, self.training)
        output = torch.cat((input_forward, input_backward), dim=-1)
        hidden = torch.cat((hidden_forward, hidden_backward), dim=-1)
        return output, hidden


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CNNClassifier(nn.Module):
    def __init__(self, input_dim, output_channel, filter, output_dim, dropout):
        super(CNNClassifier, self).__init__()

        self.cnn = nn.ModuleList([nn.Conv2d(1, output_channel, (f, input_dim)) for f in filter])

        linear_dim = output_channel * len(filter)
        self.layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(linear_dim, output_dim, bias=False),
            nn.LeakyReLU(0.1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.contiguous().unsqueeze(1)
        conv = [F.relu(cnn_(input)).squeeze(3) for cnn_ in self.cnn]
        conv = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv]
        return self.layer(torch.cat(conv, 1))


class SelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens, ent_mask=None):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        # if ent_mask is not None:
        #     for ii in range(scores.size(0)):
        #         if 1 in scores[ii]:
        #             scores[ii].masked_fill((ent_mask[ii] == 2), -np.inf)
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context


class MLPSelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, d_out, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, mask):
        batch_size, seq_len, d_feat, domains = inp.size()
        inp = self.dropout(inp)
        scores_ = self.scorer(inp.contiguous().view(batch_size, seq_len, -1))
        scores_ = scores_.masked_fill((mask == 0).unsqueeze(-1), -1e9)
        scores = F.softmax(scores_, dim=-1)
        context = scores.unsqueeze(-2).expand_as(inp).mul(inp).sum(-1)
        return context, scores_


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, n_token, hidden_size, num_layers, num_heads, dropout_prob, embedding):
        super(TransformerModel, self).__init__()

        self.d_model = hidden_size
        self.pos_encoder = PositionalEncoding(hidden_size, dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout_prob, batch_first=True),
            num_layers)
        self.encoder = embedding
        self.src_mask = None
        self.pad_mask = None
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, num_heads, hidden_size, dropout_prob, batch_first=True),
            num_layers)

        self.decoder = nn.Linear(hidden_size, n_token)
        self.init_weights()

        self.trans = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
        )
        self.relu = nn.LeakyReLU(0.1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, refer, inputs, has_mask=True):
        if has_mask:
            if self.src_mask is None or self.src_mask.size(1) != inputs.size(1):
                mask = self._generate_square_subsequent_mask(inputs.size(1))
                self.src_mask = _cuda(mask)
            self.pad_mask = _cuda((inputs == PAD_token))
            refer_mask = _cuda(self._generate_square_subsequent_mask(refer.size(1)))
            refer_pad_mask = _cuda((refer == PAD_token))
        else:
            self.src_mask = None
            self.pad_mask = None
            refer_mask = None
            refer_pad_mask = None

        refer = self.encoder(refer) * math.sqrt(self.d_model)
        refer = self.pos_encoder(refer)
        output_en = self.transformer_encoder(refer, refer_mask, refer_pad_mask)

        inputs = self.encoder(inputs) * math.sqrt(self.d_model)
        inputs = self.pos_encoder(inputs)
        output_de = self.transformer_decoder(inputs, output_en, self.src_mask, tgt_key_padding_mask=self.pad_mask)

        return output_de


class ContextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, domains, n_layers=args['layer_r']):
        super(ContextEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.domains = domains
        self.mix_attention = MLPSelfAttention(len(domains) * 2 * self.hidden_size, len(domains), dropout)
        self.mix_attention_sket = MLPSelfAttention(len(domains) * 2 * self.hidden_size, len(domains), dropout)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, args['embeddings_dim'], padding_idx=PAD_token)
        self.odim = args['embeddings_dim']
        self.global_gru = RNN_Residual(self.odim, hidden_size, n_layers, dropout=dropout)
        self.sketch_gru = RNN_Residual(self.odim, hidden_size, n_layers, dropout=dropout)

        self.sketch_resp_rnn = nn.GRU(hidden_size, hidden_size, dropout=dropout, batch_first=True)

        self.selfatten = SelfAttention(1 * self.hidden_size, dropout=self.dropout)
        self.selfatten_sket = SelfAttention(1 * self.hidden_size, dropout=self.dropout)

        for domain in domains.keys():
            setattr(self, '{}_gru'.format(domain),
                    RNN_Residual(self.odim, hidden_size, n_layers, dropout=self.dropout))
        for domain in domains.keys():
            setattr(self, '{}_gru_sketch'.format(domain),
                    RNN_Residual(self.odim, hidden_size, n_layers, dropout=self.dropout))

        self.MLP_H = nn.Sequential(
            nn.Linear(4 * self.hidden_size, 2 * self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(2 * self.hidden_size, 1 * self.hidden_size),
        )
        self.MLP_sket = nn.Sequential(
            nn.Linear(4 * self.hidden_size, 2 * self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(2 * self.hidden_size, 1 * self.hidden_size),
        )
        self.W_hid = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 1 * self.hidden_size),
            # nn.LeakyReLU(0.1),
            # nn.Linear(1 * self.hidden_size, 1 * self.hidden_size),
        )

        self.W = nn.Linear(hidden_size, 1)
        self.global_classifier = nn.Sequential(
            GradientReversal(),
            CNNClassifier(2 * hidden_size, hidden_size, [2, 3], len(domains), dropout))
        self.tfModel = TransformerModel(self.input_size, hidden_size, 4, 8, dropout, self.embedding)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def get_embedding(self, input_seqs, word_lens):
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        embedded = embedded.view(input_seqs.size() + (embedded.size(-1),))
        # embedded = torch.sum(embedded, 2).squeeze(2)
        embedded = embedded.transpose(0, 1)
        word_lens = word_lens.unsqueeze(-1)
        embedded = embedded.masked_fill(word_lens == 2, 0)
        embedded = torch.sum(embedded, 2)
        embedded = self.dropout_layer(embedded)
        return embedded

    def forward(self, input_seqs, input_lengths, sket_input_seqs, sket_input_lens, ent_mask, conv_word_lens):
        embedded_sket = self.get_embedding(sket_input_seqs, conv_word_lens)
        outputs_sket, sketch_hidden = self.sketch_gru(embedded_sket, sket_input_lens)

        local_sket_outputs = []
        mask = _cuda(torch.zeros((len(sket_input_lens), max(sket_input_lens))))
        for i, length in enumerate(sket_input_lens):
            mask[i, :length] = 1

        for domain in self.domains:
            local_rnn = getattr(self, '{}_gru_sketch'.format(domain))
            local_output, _ = local_rnn(embedded_sket, sket_input_lens)
            local_sket_outputs.append(local_output)
        local_skt_outputs, _ = self.mix_attention_sket(torch.stack(local_sket_outputs, dim=-1), mask)

        outputs_sketch = self.MLP_sket(torch.cat((F.dropout(local_skt_outputs, self.dropout, self.training),
                                                  F.dropout(outputs_sket, self.dropout, self.training)), dim=-1))
        sket_hidden = self.selfatten_sket(outputs_sketch, sket_input_lens)

        # try to encode sketch resp again
        # sketch_hidden = self.W_hid(sketch_hidden)
        sket_resp_outputs, resp_hidden = self.sketch_resp_rnn(outputs_sketch, sket_hidden.unsqueeze(0))
        resp_hidden = self.selfatten_sket(sket_resp_outputs, sket_input_lens)

        resp_hidden = resp_hidden + sket_hidden
        sket_resp_outputs = sket_resp_outputs + outputs_sketch

        # bla bla bla

        embedded = self.get_embedding(input_seqs, conv_word_lens)
        global_outputs, global_hidden = self.global_gru(embedded, input_lengths)

        local_outputs = []
        mask = _cuda(torch.zeros((len(input_lengths), input_lengths[0])))
        for i, length in enumerate(input_lengths):
            mask[i, :length] = 1

        for domain in self.domains:
            local_rnn = getattr(self, '{}_gru'.format(domain))
            local_output, _ = local_rnn(embedded, input_lengths)
            local_outputs.append(local_output)
        local_outputs, _ = self.mix_attention(torch.stack(local_outputs, dim=-1), mask)

        outputs_ = self.MLP_H(torch.cat((F.dropout(local_outputs, self.dropout, self.training),
                                         F.dropout(global_outputs, self.dropout, self.training)), dim=-1))

        hidden_ = self.selfatten(outputs_, input_lengths, ent_mask)
        # label = self.global_classifier(global_outputs)

        return outputs_, hidden_, None, None, sket_resp_outputs, resp_hidden


class ExternalKnowledge(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)

        self.MLP_concat_embed = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, 1 * self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(1 * self.embedding_dim, 1 * self.embedding_dim),
        )
        self.MLP_concat_embed_c = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, 1 * self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(1 * self.embedding_dim, 1 * self.embedding_dim),
        )
        self.relu = nn.LeakyReLU(0.1)
        self.fused = nn.Sequential(
            # nn.Linear(3 * self.embedding_dim, 2 * self.embedding_dim),
            # nn.LeakyReLU(0.1),
            nn.Linear(2 * self.embedding_dim, 1 * self.embedding_dim),
        )
        self.fused_kb = nn.Sequential(
            nn.Linear(self.max_hops * self.embedding_dim, 1 * self.embedding_dim),
            # nn.LeakyReLU(0.1),
            # nn.Linear(2 * self.embedding_dim, 1 * self.embedding_dim),
        )
        self.fused_kb_output = nn.Sequential(
            nn.Linear(self.max_hops * self.embedding_dim, 1 * self.embedding_dim),
            # nn.LeakyReLU(0.1),
            # nn.Linear(2 * self.embedding_dim, 1 * self.embedding_dim),
        )

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi], kb_len[bi] + conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def get_ck(self, hop, story, story_size, ctx_word_lens):
        embed = self.C[hop](story.contiguous().view(story_size[0], -1))
        embed = embed.view(story_size + (embed.size(-1),))
        ctx_word_lens = ctx_word_lens.unsqueeze(-1)
        embed = embed.masked_fill(ctx_word_lens == 2, 0)
        embed = torch.sum(embed, 2)
        return embed

    def get_ck_local(self, hop, story, story_size, domains):
        embed = _cuda(torch.zeros((story_size + (self.embedding_dim,))))
        for i, domain in enumerate(domains):
            embed[i] = self.__getattribute__('C_{}_'.format(domain))[hop](story.contiguous()[i])
        embed = torch.sum(embed, 2).squeeze(2)
        return embed

    def load_memory(self, story, kb_len, conv_len, hidden, dh_outputs, domains, hidden_fine, dh_outputs_fine, ctx_word_lens):
        # Forward multiple hop mechanism
        # hidden = self.relu(self.fused(hidden))
        u = [hidden.squeeze(0)]
        story_size = story.size()
        kb_outputs = []
        # self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.get_ck(hop, story, story_size, ctx_word_lens)
            embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs)
            embed_A = self.dropout_layer(embed_A)

            if len(list(u[-1].size())) == 1:
                u[-1] = u[-1].unsqueeze(0)
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A * u_temp, 2)
            prob_ = self.softmax(prob_logit)

            embed_C = self.get_ck(hop + 1, story, story_size, ctx_word_lens)
            embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs)

            prob = prob_.unsqueeze(2).expand_as(embed_C)

            kb_outputs.append(prob * embed_C)

            o_k = torch.sum(embed_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
            # self.m_story.append(embed_A)
        # self.m_story.append(embed_C)
        # print(kb_emb.shape)
        kb_output = self.relu(self.fused_kb_output(torch.cat(kb_outputs, dim=-1)))

        ent_pointer, kb_emb, kb_ent_hdd = self.load_ent_memory(story, kb_len, conv_len, hidden, dh_outputs, domains, hidden_fine, dh_outputs_fine, ctx_word_lens)
        return self.sigmoid(prob_logit), u[-1], ent_pointer, kb_emb, kb_output, kb_ent_hdd

    def load_ent_memory(self, story, kb_len, conv_len, hidden, dh_outputs, domains, hidden_fine, dh_outputs_fine, ctx_word_lens):
        u_ent = [hidden_fine.squeeze(0)]
        story_size = story.size()
        kb_embs = []
        self.m_story_ent = []
        for hop in range(self.max_hops):
            embed_A = self.get_ck(hop, story, story_size, ctx_word_lens)
            embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs_fine)
            embed_A = self.dropout_layer(embed_A)

            if len(list(u_ent[-1].size())) == 1:
                u_ent[-1] = u_ent[-1].unsqueeze(0)
            u_temp = u_ent[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A * u_temp, 2)
            prob_ = self.softmax(prob_logit)

            embed_C = self.get_ck(hop + 1, story, story_size, ctx_word_lens)
            embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs_fine)

            prob = prob_.unsqueeze(2).expand_as(embed_C)

            kb_embs.append(embed_C * prob)

            o_k = torch.sum(embed_C * prob, 1)
            u_k = u_ent[-1] + o_k
            u_ent.append(u_k)
            self.m_story_ent.append(embed_A)
        self.m_story_ent.append(embed_C)

        kb_emb = self.relu(self.fused_kb(torch.cat(kb_embs, dim=-1)))
        return self.sigmoid(prob_logit), kb_emb, u_ent[-1]

    def forward(self, query_vector, global_pointer):
        u = [query_vector]
        for hop in range(self.max_hops):
            m_A = self.m_story_ent[hop]
            m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)
            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A * u_temp, 2)
            prob_soft = self.softmax(prob_logits)

            m_C = self.m_story_ent[hop + 1]
            m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return prob_soft, prob_logits


class LocalMemoryDecoder(nn.Module):
    def __init__(self, shared_emb, lang, hidden_dim, hop, dropout, domains=None):
        super(LocalMemoryDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.max_hops = hop
        self.C = shared_emb
        self.embedding_dim = shared_emb.embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.domains = domains

        self.sketch_rnn_global = nn.GRU(self.embedding_dim, hidden_dim, dropout=dropout)
        for index, domain in enumerate(domains):
            local = nn.GRU(self.embedding_dim, hidden_dim, dropout=dropout)
            self.add_module("sketch_rnn_local_{}".format(index), local)
        self.sketch_rnn_local = AttrProxy(self, "sketch_rnn_local_")
        self.mix_attention = MLPSelfAttention(len(domains) * hidden_dim, len(domains), dropout)
        self.relu = nn.ReLU()
        self.projector = nn.Sequential(
            # nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            # nn.LeakyReLU(0.1),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.MLP = nn.Sequential(
            nn.Linear(2 * hidden_dim, 1 * hidden_dim),
            # nn.LeakyReLU(0.1),
            # nn.Linear(1 * hidden_dim, hidden_dim),
            # nn.LeakyReLU(0.1),
        )

        self.attn_table = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.attn_table_fine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.attn_kb_ent = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.projector2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.projector3 = nn.Sequential(
            # nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            # nn.Tanh(),
            nn.Linear(3 * hidden_dim, hidden_dim),
        )
        self.projector4 = nn.Sequential(
            # nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            # nn.Tanh(),
            nn.Linear(3 * hidden_dim, hidden_dim),
        )
        self.domain_emb = nn.Embedding(len(domains), self.embedding_dim)

        self.global_classifier = nn.Sequential(GradientReversal(),
                                               CNNClassifier(hidden_dim, hidden_dim, [2, 3], len(domains), dropout))

    def get_p_vocab(self, hidden, H):
        cond = self.attn_table(torch.cat((H, hidden.unsqueeze(1).expand_as(H)), dim=-1))
        cond = F.softmax(cond.squeeze(-1), dim=-1)
        hidden_ = cond.unsqueeze(-1).expand_as(H).mul(H).sum(-2)
        context = torch.tanh(self.projector2(torch.cat((hidden, hidden_), dim=-1).unsqueeze(0)))
        p_vocab = self.attend_vocab(self.C.weight, context.squeeze(0))
        return p_vocab, context

    def get_p_vocab_atten(self, hidden, H, outputs, kb_output):
        h = hidden.unsqueeze(1)
        atten_weights = self.attn_table(torch.cat((H, h.expand_as(H)), dim=-1))
        atten_weights = F.softmax(atten_weights.transpose(1, 2), dim=-1)
        sket_hidden = atten_weights.bmm(H)

        # atten_weights1 = self.attn_table(torch.cat((outputs, h.expand_as(outputs)), dim=-1))
        # atten_weights1 = F.softmax(atten_weights1.transpose(1, 2), dim=-1)
        # out = atten_weights1.bmm(outputs)

        atten_weights1 = self.attn_table_fine(torch.cat((kb_output, h.expand_as(kb_output)), dim=-1))
        atten_weights1 = F.softmax(atten_weights1.transpose(1, 2), dim=-1)
        kb_hiddn = atten_weights1.bmm(kb_output)

        context = torch.tanh(self.projector4(torch.cat((sket_hidden, h, kb_hiddn), dim=-1))).transpose(0, 1)
        p_vocab = self.attend_vocab(self.C.weight, context.squeeze(0))
        return p_vocab, context

    def fused_output(self, hdd, outputs, kb_readout):
        h = hdd.transpose(0, 1)
        atten_weights = self.attn_kb_ent(torch.cat((outputs, h.expand_as(outputs)), dim=-1))
        atten_weights = F.softmax(atten_weights.transpose(1, 2), dim=-1)
        out_h = atten_weights.bmm(outputs)

        context = torch.tanh(self.projector3(torch.cat((out_h, h, kb_readout.unsqueeze(1)), dim=-1))).squeeze(1)
        return context

    def forward(self, extKnow, story_size, story_lengths, copy_list, encode_hidden, target_batches, max_target_length,
                batch_size, use_teacher_forcing, get_decoded_words, global_pointer, H=None, global_entity_type=None,
                domains=None, kb_readout=None, outputs=None, kb_emb=None, kb_output=None):
        # Initialize variables for vocab and pointer
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, story_size[1]))

        decoder_input = self.C(_cuda(torch.LongTensor([SOS_token] * batch_size)))
        memory_mask_for_step = _cuda(torch.ones(story_size[0], story_size[1]))
        decoded_fine, decoded_coarse = [], []

        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)
        # hidden = encode_hidden.unsqueeze(0)

        hidden_locals = []
        for i in range(len(self.domains)):
            hidden_locals.append(hidden.clone())

        mask = _cuda(torch.ones((len(story_lengths), 1)))
        global_hiddens = []
        local_hiddens = []
        scores = []

        # Start to generate word-by-word
        for t in range(max_target_length):
            if t != 0:
                decoder_input = self.C(decoder_input)
            embed_q = self.dropout_layer(decoder_input)
            embed_q = embed_q.view(1, -1, self.embedding_dim)
            gl_output, hidden = self.sketch_rnn_global(embed_q, hidden)

            hidden_locals_ = []
            for domain in self.domains.values():
                hidden_locals_.append(self.sketch_rnn_local[domain](embed_q, hidden_locals[domain])[1])
            hidden_locals = hidden_locals_
            hidden_local, score = self.mix_attention(torch.stack(hidden_locals, dim=-1).transpose(0, 1),
                                                     mask)
            hidden_local, score = hidden_local.transpose(0, 1), score.transpose(0, 1)
            scores.append(score)
            query_vector = self.MLP(torch.cat((F.dropout(hidden, self.dropout, self.training),
                                               F.dropout(hidden_local, self.dropout, self.training)), dim=-1))
            global_hiddens.append(query_vector)
            # global_hiddens.append(hidden)
            local_hiddens.append(hidden_local)

            p_vocab, context = self.get_p_vocab_atten(query_vector[0], H, outputs, kb_output)
            # p_vocab, context = self.get_p_vocab(query_vector[0], H)

            all_decoder_outputs_vocab[t] = p_vocab
            _, topvi = p_vocab.data.topk(1)

            if use_teacher_forcing:
                decoder_input = target_batches[:, t]
            else:
                decoder_input = topvi.squeeze()

            # context = torch.tanh(self.projector3(torch.cat((context[0], kb_readout), dim=-1)))
            context = self.fused_output(context, kb_emb, kb_readout)
            # query the external konwledge using the hidden state of sketch RNN
            prob_soft, prob_logits = extKnow(context, global_pointer)
            all_decoder_outputs_ptr[t] = prob_logits

            if get_decoded_words:

                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)
                temp_f, temp_c = [], []

                for bi in range(batch_size):
                    token = topvi[bi].item()
                    temp_c.append(self.lang.index2word[token])

                    if '@' in self.lang.index2word[token]:
                        gold_type = self.lang.index2word[token]
                        cw = 'UNK'
                        for i in range(search_len):
                            if toppi[:, i][bi] < story_lengths[bi] - 1:
                                cw = copy_list[bi][toppi[:, i][bi].item()]
                                break
                        temp_f.append(cw)

                        if args['record']:
                            memory_mask_for_step[bi, toppi[:, i][bi].item()] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])

                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

        # label = self.global_classifier(torch.cat(global_hiddens, dim=0).transpose(0, 1))
        # scores = torch.cat(scores, dim=0).transpose(0, 1).contiguous()
        # print('all_decoder_outputs_vocab: ', all_decoder_outputs_vocab.shape)
        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, decoded_fine, decoded_coarse, None, None

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        return scores_


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
