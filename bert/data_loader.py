import os
import torch
import numpy as np
import json
import re
from torch.autograd import Variable


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq(seq_a, max_length):
    while len(seq_a) > max_length:
        seq_a.pop()


def _truncate_ent_list(ent_list, max_length):
    """Truncates or pads entity list"""

    if len(ent_list) > max_length:
        ent_list = ent_list[:max_length]
    elif len(ent_list) < max_length:
        ent_list += [0] * (max_length - len(ent_list))
    return ent_list


def _truncate_sentence(sentence, max_length, is_query):
    """
    Truncates or pads natural language sentences
    Use different padding for queries and passages ([query_padding] and [passage_padding]
    """
    re.sub('[.,?!:;\'\"]', '', sentence)
    sentence = sentence.lower().strip().split(' ')
    if len(sentence) > max_length:
        sentence = sentence[:max_length]
    elif len(sentence) < max_length:
        if is_query:
            padding = [' [query_padding]'] * (max_length - len(sentence))
        else:
            padding = [' [passage_padding]'] * (max_length - len(sentence))
        sentence += padding
    assert len(sentence) == max_length
    return sentence


def tok2int_sent(sentence, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    # sent_a, title, sent_b = sentence
    # sent_a, sent_b = sentence
    sent_a = sentence
    tokens_a = tokenizer.tokenize(sent_a)
    # TODO: use arg to replace magic number
    len_a = len(tokens_a)
    _truncate_seq(tokens_a, max_seq_length - 2)


    tokens_b = None
    # if sent_b:
    #     # tokens_t = tokenizer.tokenize(title)
    #     tokens_b = tokenizer.tokenize(sent_b)
    #     # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    #     _truncate_seq(tokens_b, max_seq_length - 3 - 15)
    # else:
    #     # Account for [CLS] and [SEP] with "- 2"
    #     if len(tokens_a) > max_seq_length - 2:
    #         tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    # segment_ids = [0] * 17
    segment_ids = [1] * len(tokens)
    if tokens_b:
        # tokens = tokens + tokens_t + ["[SEP]"] + tokens_b + ["[SEP]"]
        # segment_ids += [1] * (len(tokens_b) + len(tokens_t) + 2)
        tokens = tokens + tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # input_ids = input_ids[:len_a + 1] + [0] * (15 - len_a) + input_ids[len_a + 1:]
    # input_mask = [1] * (len_a + 1) + [0] * (15 - len_a) + [1] * (len(input_ids) - 16)
    input_mask = [1] * len(tokens)
    padding = [0] * (max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def tok2int_list(src_list, tokenizer, max_seq_length, max_seq_size=-1):
    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    for step, sent in enumerate(src_list):
        input_ids, input_mask, input_seg = tok2int_sent(sent, tokenizer, max_seq_length)
        inp_padding.append(input_ids)
        msk_padding.append(input_mask)
        seg_padding.append(input_seg)
    #if max_seq_size != -1:
    #    inp_padding = inp_padding[:max_seq_size]
    #    msk_padding = msk_padding[:max_seq_size]
    #    seg_padding = seg_padding[:max_seq_size]
    #    inp_padding += ([[0] * max_seq_length] * (max_seq_size - len(inp_padding)))
    #    msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(msk_padding)))
    #    seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(seg_padding)))
    return inp_padding, msk_padding, seg_padding


def process_sent(sentence):
    sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
    sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
    sentence = re.sub(" -LRB-", " ( ", sentence)
    sentence = re.sub("-RRB-", " )", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    return sentence


def process_wiki_title(title):
    title = re.sub("_", " ", title)
    title = re.sub(" -LRB-", " ( ", title)
    title = re.sub("-RRB-", " )", title)
    title = re.sub("-COLON-", ":", title)
    return title


def get_matrix(query, passage):
    matrix = list()
    for p in passage:
        row = list()
        for q in query:
            row.append(1) if p == q else row.append(0)
        matrix.append(row)
    return matrix


class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, test=False, cuda=True, batch_size=64):
        self.cuda = cuda
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.threshold = args.threshold
        self.data_path = data_path + "triples"
        self.ent_path = data_path + "ent_triples"
        self.test = test
        self.id = 0
        self.list = self.get_file(self.data_path)
        self.examples, self.entity = self.read_file(self.data_path, self.ent_path)
        self.total_num = len(self.examples)
        if self.test:
            self.total_num = 100000
            self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
            self.shuffle()
        else:
            self.total_step = self.total_num / batch_size
            self.shuffle()
        self.step = 0

    def get_file(self, data_path):
        file_list = list()
        with open(data_path, 'r') as f:
            for line in f:
                file_list.append(line.strip().split('\t'))
        return file_list

    def read_file(self, data_path, ent_path):
        entities = list()
        with open(ent_path) as f_in:
            for step, line in enumerate(f_in):
                sublines = line.strip().split('\t')
                if len(sublines) != 3:
                    continue
                qids = sublines[0].split(',')[:-1]
                pos_pids = sublines[1].split(',')[:-1]
                neg_pids = sublines[2].split(',')[:-1]
                for i in range(len(qids)):
                    qids[i] = int(qids[i]) + 1
                for i in range(len(pos_pids)):
                    pos_pids[i] = int(pos_pids[i]) + 1
                for i in range(len(neg_pids)):
                    neg_pids[i] = int(neg_pids[i]) + 1
                qids = _truncate_ent_list(qids, 2)
                pos_pids = _truncate_ent_list(pos_pids, 12)
                neg_pids = _truncate_ent_list(neg_pids, 12)
                entities.append([qids, pos_pids, neg_pids])

        examples = list()   # examples: [query, pos_passage, neg_passage]
        with open(data_path) as f_in:
            for step, line in enumerate(f_in):
                sublines = line.strip().split("\t")
                if len(sublines) != 3:
                    continue
                examples.append([process_sent(sublines[0]), process_sent(sublines[1]), process_sent(sublines[2])])

        assert len(entities) == len(examples)
        return examples, entities

    def shuffle(self):
        # TODO: shuffle?
        pass
        # np.random.shuffle(self.examples)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''
        if self.step < self.total_step:
            # examples: a batch of input
            examples = self.examples[self.step * self.batch_size:(self.step + 1) * self.batch_size]
            entities = self.entity[self.step * self.batch_size:(self.step + 1) * self.batch_size]

            # TODO: process query and passage in examples and get interaction matrix for local model
            pos_matrix = list()
            neg_matrix = list()
            for example in examples:
                query = _truncate_sentence(example[0], 10, True)
                pos_psg = _truncate_sentence(example[1], 60, False)
                neg_psg = _truncate_sentence(example[2], 60, False)
                pos_matrix.append(get_matrix(query, pos_psg))
                neg_matrix.append(get_matrix(query, neg_psg))

            pos_inputs = list()
            neg_inputs = list()
            qry_inputs = list()
            for example in examples:
                # pos_inputs.append([example[0], example[1]])
                # neg_inputs.append([example[0], example[2]])
                pos_inputs.append(example[1])
                neg_inputs.append(example[2])
                qry_inputs.append(example[0])
            inp_pos, msk_pos, seg_pos = tok2int_list(pos_inputs, self.tokenizer, 80)
            inp_neg, msk_neg, seg_neg = tok2int_list(neg_inputs, self.tokenizer, 80)
            inp_qry, msk_qry, seg_qry = tok2int_list(qry_inputs, self.tokenizer, 15)

            inp_tensor_pos = torch.Tensor(inp_pos).long()
            msk_tensor_pos = torch.Tensor(msk_pos).long()
            seg_tensor_pos = torch.Tensor(seg_pos).long()
            inp_tensor_neg = torch.Tensor(inp_neg).long()
            msk_tensor_neg = torch.Tensor(msk_neg).long()
            seg_tensor_neg = torch.Tensor(seg_neg).long()
            inp_tensor_qry = torch.Tensor(inp_qry).long()
            msk_tensor_qry = torch.Tensor(msk_qry).long()
            seg_tensor_qry = torch.Tensor(seg_qry).long()
            pos_matrix = torch.Tensor(pos_matrix).float()
            neg_matrix = torch.Tensor(neg_matrix).float()

            ent_tensor_query = list()
            ent_tensor_pos = list()
            ent_tensor_neg = list()
            for e in entities:
                ent_tensor_query.append(e[0])
                ent_tensor_pos.append(e[1])
                ent_tensor_neg.append(e[2])
            ent_tensor_query = torch.Tensor(ent_tensor_query).long()
            ent_tensor_pos = torch.Tensor(ent_tensor_pos).long()
            ent_tensor_neg = torch.Tensor(ent_tensor_neg).long()

            if self.cuda:
                inp_tensor_pos = inp_tensor_pos.cuda()
                msk_tensor_pos = msk_tensor_pos.cuda()
                seg_tensor_pos = seg_tensor_pos.cuda()
                inp_tensor_neg = inp_tensor_neg.cuda()
                msk_tensor_neg = msk_tensor_neg.cuda()
                seg_tensor_neg = seg_tensor_neg.cuda()
                inp_tensor_qry = inp_tensor_qry.cuda()
                msk_tensor_qry = msk_tensor_qry.cuda()
                seg_tensor_qry = seg_tensor_qry.cuda()
                ent_tensor_query = ent_tensor_query.cuda()
                ent_tensor_pos = ent_tensor_pos.cuda()
                ent_tensor_neg = ent_tensor_neg.cuda()
                pos_matrix = pos_matrix.cuda()
                neg_matrix = neg_matrix.cuda()
            inp_ent_pos = [ent_tensor_query, ent_tensor_pos]
            inp_ent_neg = [ent_tensor_query, ent_tensor_neg]
            self.step += 1
            return inp_tensor_qry, msk_tensor_qry, seg_tensor_qry, inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg,\
                inp_ent_pos, inp_ent_neg, pos_matrix, neg_matrix
        else:
            self.step = 0
            self.id = 0
            if not self.test:
                # examples = self.read_file(self.data_path)
                # self.examples = examples
                self.shuffle()
            raise StopIteration()


class DataLoaderTest(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, cuda=True, batch_size=64):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.threshold = args.threshold
        self.data_path = data_path
        inputs, ids, evi_list = self.read_file(data_path)
        self.inputs = inputs
        self.ids = ids
        self.evi_list = evi_list

        self.total_num = len(inputs)
        self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        self.step = 0

    def read_file(self, data_path):
        inputs = list()
        ids = list()
        evi_list = list()
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                instance = json.loads(line.strip())
                claim = instance['claim']
                id = instance['id']
                for evidence in instance['evidence']:
                    ids.append(id)
                    inputs.append([process_sent(claim), process_wiki_title(evidence[0]), process_sent(evidence[2])])
                    evi_list.append(evidence)
        return inputs, ids, evi_list

    def shuffle(self):
        np.random.shuffle(self.examples)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''
        if self.step < self.total_step:
            inputs = self.inputs[self.step * self.batch_size : (self.step+1)*self.batch_size]
            ids = self.ids[self.step * self.batch_size: (self.step + 1) * self.batch_size]
            evi_list = self.evi_list[self.step * self.batch_size: (self.step + 1) * self.batch_size]
            inp, msk, seg = tok2int_list(inputs, self.tokenizer, self.max_len, -1)
            inp_tensor_input = Variable(
                torch.LongTensor(inp))
            msk_tensor_input = Variable(
                torch.LongTensor(msk))
            seg_tensor_input = Variable(
                torch.LongTensor(seg))
            if self.cuda:
                inp_tensor_input = inp_tensor_input.cuda()
                msk_tensor_input = msk_tensor_input.cuda()
                seg_tensor_input = seg_tensor_input.cuda()
            self.step += 1
            return inp_tensor_input, msk_tensor_input, seg_tensor_input, ids, evi_list
        else:
            self.step = 0
            raise StopIteration()
