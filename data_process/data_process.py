import re
import os
import sys
import json
import argparse
import copy
import numpy as np
from utils import Type
import random
p2id_path = "../../raw_data/passage2id/collection.tsv"
q2id_path = "../../raw_data/query2id/queries.train.tsv"
q2id_dev_path = "../../raw_data/query2id/queries.dev.tsv"
used_pid = "../../raw_data/passage2id/ent/id_used.txt"
used_qid = "../../raw_data/query2id/ent/id_used.txt"
train_path = "../../raw_data/train/qidpidtriples.train.full.tsv"
eval_path = "../../raw_data/top1000_eval/top1000.eval"
dev_path = "../../raw_data/top1000_dev/id_dev"
output_path = "../../raw_data/train/triples/"
ent_map_path = "../../raw_data/entity/entity_map.txt"
ent2id_path = "../../raw_data/entity/entity2id.txt"
ent_out_path = "../../raw_data/entity/ent2id"
p_ent_path = "../../raw_data/passage2id/ent/ent_passage"
q_ent_path = "../../raw_data/query2id/ent/ent_query"


def generate_train_data():
    """
    Get training data. Each line contains
    triples: query\tpos_passage\tneg_passage\t
    ent_triples: q_ent_id,...,query_ent_id\tpos_p_ent_id,...pos_p_ent_id\tneg_p_ent_id,...,neg_p_ent_id\t

    Lines with no query entities or less than 6 passage entities are disputed.
    """
    passages = dict()
    queries = dict()
    with open(p2id_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            passages[int(tokens[0])] = tokens[1]
    print("read passages")
    with open(q2id_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            queries[int(tokens[0])] = tokens[1]
    print("read queries")
    q_ent = dict()
    p_ent = dict()
    with open(q_ent_path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                print(line)
            q_ent[obj['id']] = obj['passage']
    with open(p_ent_path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                print(line)
            p_ent[obj['id']] = obj['passage']

    count = 0
    index = 0
    f_out = open(output_path + 'triples' + str(index), 'w')
    f_ent_out = open(output_path + 'ent_triples' + str(index), 'w')
    with open(train_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 3:
                continue
            count += 1
            if int(tokens[0]) not in queries or int(tokens[1]) not in passages or int(tokens[2]) not in passages\
                    or tokens[0] not in q_ent or tokens[1] not in p_ent or tokens[2] not in p_ent:
                continue
            if len(q_ent[tokens[0]]) == 0 or len(p_ent[tokens[1]]) < 6 or len(p_ent[tokens[2]]) < 6:
                continue
            f_out.write(queries[int(tokens[0])] + '\t' +
                        passages[int(tokens[1])] + '\t' + passages[int(tokens[2])] + '\n')
            for e in q_ent[tokens[0]]:
                f_ent_out.write(str(e) + ',')
            f_ent_out.write('\t')
            for e in p_ent[tokens[1]]:
                f_ent_out.write(str(e) + ',')
            f_ent_out.write('\t')
            for e in p_ent[tokens[2]]:
                f_ent_out.write(str(e) + ',')
            f_ent_out.write('\n')
            if count >= 2000000:
                print("finish generation: %d" % index)
                count = 0
                index += 1
                f_out.close()
                f_ent_out.close()
                if index >= 10:
                    break
                f_out = open(output_path + 'triples' + str(index), 'w')
                f_ent_out = open(output_path + 'ent_triples' + str(index), 'w')
    f_out.close()
    f_ent_out.close()


def generate_dev_data(path, out_path):
    qids = dict()
    sorted_qids = set()
    with open(path) as f:
        for step, line in enumerate(f):
            tokens = line.strip().split('\t')
            if len(tokens) != 4:
                continue
            qid = int(tokens[0])
            pid = tokens[1]
            sorted_qids.add(qid)
            if qid not in qids:
                qids[qid] = [pid]
            else:
                qids[qid].append(pid)
    with open(out_path, 'w') as f:
        for id in sorted_qids:
            f.write(str(id) + '\t')
            for pid in qids[id]:
                f.write(pid + '\t')
            f.write('\n')

    print("total queries: %d " % len(qids))


def gen_eval_data(out_path, ent_out_path):
    passages = dict()
    queries = dict()
    with open(p2id_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            passages[int(tokens[0])] = tokens[1]
    print("read passages")
    with open(q2id_dev_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            queries[int(tokens[0])] = tokens[1]
    print("read queries")
    q_ent = dict()
    p_ent = dict()
    with open(q_ent_path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                print(line)
            q_ent[obj['id']] = obj['passage']
    print("read query id")
    with open(p_ent_path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                print(line)
            p_ent[obj['id']] = obj['passage']
    print("read passage id")
    f_out = open(out_path, 'w')
    f_ent_out = open(ent_out_path, 'w')
    i = 0
    with open(dev_path, 'r') as f:
        for line in f:
            i += 1

            subline = line.strip().split('\t')
            qid = subline[0]
            pids = subline[1:]
            if qid not in queries or qid not in q_ent:
                print("query %s missing!" % qid)
                continue
            f_out.write(queries[qid] + '\t')
            for e in q_ent[qid]:
                f_ent_out.write(str(e) + ',')
            f_ent_out.write('\t')
            for pid in pids:
                if int(pid) not in passages or int(pid) not in p_ent:
                    continue
                f_out.write(passages[pid] + '\t')
                for e in p_ent[pid]:
                    f_ent_out.write(str(e) + ',')
                f_ent_out.write('\t')
            f_out.write('\n')
            f_ent_out.write('\n')


def divide_data():
    for i in range(1, 2):
        with open(output_path + str(i), 'r') as f:
            j = 1
            count = 0
            f_out = open(output_path + str(i) + str(j), 'w')
            for line in f:
                f_out.write(line)
                count += 1
                if count >= 2000000:
                    count = 0
                    j += 1
                    f_out.close()
                    f_out = open(output_path + str(i) + str(j), 'w')
            f_out.close()


def get_emb_id():
    """
    Generates ent2id using entity_map.txt and entity2id.txt.
    Format of each line: entity \t id
    """
    entities = dict()
    with open(ent2id_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            entities[tokens[0]] = int(tokens[1])
    with open(ent_out_path, 'w') as f_out:
        with open(ent_map_path, 'r') as f:
            for line in f:
                tokens = line.strip().split('\t')
                if len(tokens) != 2:
                    continue
                if tokens[1] in entities:
                    f_out.write(tokens[0] + '\t' + str(entities[tokens[1]]) + '\n')


def get_used_id():
    """
    Generates ids of passages and queries used in train, dev and dval data
    """
    pid = set()
    qid = set()
    with open(train_path, 'r') as f:
        i = 0
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 3:
                continue
            qid.add(int(tokens[0]))
            pid.add(int(tokens[1]))
            pid.add(int(tokens[2]))
            i += 1
            if i > 2000000:
                break
    with open(dev_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 4:
                continue
            qid.add(int(tokens[0]))
            pid.add(int(tokens[1]))
    with open(eval_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 4:
                continue
            qid.add(int(tokens[0]))
            pid.add(int(tokens[1]))

    print("passages: %d" % len(pid))
    print("queries: %d" % len(qid))
    with open(used_qid, 'w') as f_out:
        with open(q2id_path) as f:
            for line in f:
                tokens = line.strip().split('\t')
                if len(tokens) != 2:
                    continue
                if int(tokens[0]) in qid:
                    f_out.write(line)
    with open(used_pid, 'w') as f_out:
        with open(p2id_path) as f:
            for line in f:
                tokens = line.strip().split('\t')
                if len(tokens) != 2:
                    continue
                if int(tokens[0]) in pid:
                    f_out.write(line)


def join(num, path):
    with open(path, 'a') as f_out:
        for i in range(num):
            with open(path + str(i), 'r') as f:
                for line in f:
                    f_out.write(line)
            f_out.write('\n')


def get_remained():
    pid = set()
    with open("../../raw_data/passage2id/ent/ent_used.txt") as f:
        for step, line in enumerate(f):
            try:
                obj = json.loads(line)
            except:
                continue
            pid.add(obj['id'])
            if (step % 1000000) == 0:
                print(step)
    print("processed: %d" % len(pid))
    with open("../../raw_data/passage2id/ent/remain.txt", 'w') as f_out:
        with open("../../raw_data/passage2id/ent/id_used.txt", 'r') as f:
            for line in f:
                tokens = line.split('\t')
                if tokens[0] not in pid:
                    f_out.write(line)


def get_train_id():
    with open("../../raw_data/train/ent/ent_triples1", 'w') as f_out:
        with open(train_path, 'r') as f:
            i = 0
            for line in f:
                f_out.write(line)
                i += 1
                if i >= 2000000:
                    break


def deduplicate(path, out_path):
    ids = set()
    f_out = open(out_path, 'w')
    with open(path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                continue
            if obj['id'] not in ids:
                ids.add(obj['id'])
                f_out.write(line)


def count_ent(path):
    """
    Print infomation of entity number in passages or queries
    """
    numbers = list()
    sum = 0
    min = 100
    max = 0
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                pass
            count = len(obj['passage'])
            sum += count
            numbers.append(count)
            if count < min:
                min = count
            if count > max:
                max = count
    size = len(numbers)
    avg = sum / size
    numbers.sort()
    print("average: %.2f" % avg)
    mid = numbers[int(size / 2)]
    print("mid: %d" % mid)
    print("min: %d  max: %d" % (min, max))
    top = numbers[int(0.8 * size)]
    down = numbers[int(0.2 * size)]
    print("20% over {}, 20% less than {}".format(top, down))


def count_word(path):
    """
        Print infomation of passage words
    """
    numbers = list()
    sum = 0
    min = 100
    max = 0
    with open(path) as f:
        for line in f:
            passage = line.strip().split('\t')[1]
            count = len(passage.split(' '))
            sum += count
            numbers.append(count)
            if count < min:
                min = count
            if count > max:
                max = count
    size = len(numbers)
    avg = sum / size
    numbers.sort()
    print("average: %.2f" % avg)
    mid = numbers[int(size / 2)]
    print("mid: %d" % mid)
    print("min: %d  max: %d" % (min, max))
    top = numbers[int(0.8 * size)]
    down = numbers[int(0.2 * size)]
    print("20% over {}, 20% less than {}".format(top, down))


def convert_id(path, map_path):
    id_map = dict()
    with open(map_path) as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 3:
                continue
            id_map[int(tokens[1])] = int(tokens[2])
    with open(path + "_", 'w') as f_out:
        with open(path) as f:
            for step, line in enumerate(f):
                try:
                    obj = json.loads(line)
                except:
                    continue
                object = dict()
                object['id'] = obj['id']
                pids = list()
                for pid in obj['passage']:
                    if pid in id_map:
                        pids.append(id_map[pid])
                    else:
                        pass
                        # print("missed!")
                object['passage'] = pids
                object = json.dumps(object)
                f_out.write(object)
                f_out.write('\n')
                if (step % 100000) == 0:
                    print("processed:%d" % step)


def gen_test_data(path):
    with open(path + "ent_triples", 'w') as f_out:
        with open(path + "ent_triples0") as f:
            for step, line in enumerate(f):
                f_out.write(line)
                if step >= 99999:
                    break
    with open(path + "triples", 'w') as f_out:
        with open(path + "triples0") as f:
            for step, line in enumerate(f):
                f_out.write(line)
                if step >= 99999:
                    break


def count_used_entity(entity_path, out_path, passage_path, query_path):
    used_id = dict()
    with open(passage_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                continue
            for id in obj['passage']:
                if id in used_id:
                    used_id[id] += 1
                else:
                    used_id[id] = 1
    with open(query_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                continue
            for id in obj['passage']:
                if id in used_id:
                    used_id[id] += 1
                else:
                    used_id[id] = 1
    numbers = [0] * 10
    for id in used_id:
        if used_id[id] <= 10:
            numbers[used_id[id] - 1] += 1
    for step, i in enumerate(numbers):
        print("entity used %d times: %d" % (step, i))
    with open(out_path, 'w') as f_out:
        with open(entity_path, 'r') as f:
            for line in f:
                tokens = line.strip().split('\t')
                id = int(tokens[2])
                if id in used_id and used_id[id] > 4:
                    f_out.write(line)
    print("used id: %d" % len(used_id))


def process_qrels(path):
    queries = dict()
    with open(path) as f:
        for step, line in enumerate(f):
            tokens = line.strip().split('\t')
            qid = int(tokens[0])
            pid = int(tokens[2])
            if qid not in queries:
                queries[qid] = [pid]
            else:
                queries[qid].append(pid)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_robust", default=False)
    parser.add_argument("--process_cw09", default=False)
    parser.add_argument("--process_ent", default=False)
    parser.add_argument("--merge", default=False)
    parser.add_argument("--ent2id", default=False)
    parser.add_argument("--file2input", default=False)
    parser.add_argument("--sort_file", default=False)
    parser.add_argument("--get_edges", default=False)
    parser.add_argument("--complement_edges", default=False)

    # generate_train_data()
    # generate_dev_data("../../raw_data/top1000_eval/top1000.eval", "../../raw_data/top1000_eval/id_eval")
    # divide_data()
    # get_emb_id()
    # get_used_id()
    # join(2, "../../raw_data/passage2id/ent/ent_used.txt")
    # get_remained()
    # get_train_id()
    # get_triples_ent("../../raw_data/train/ent/ent_triples1", "../../raw_data/train/triples/ent_triples")
    # count_ent("../../raw_data/passage2id/ent/ent_passage")
    # count_word("../../raw_data/passage2id/id_used.txt")
    # deduplicate("../../raw_data/passage2id/ent/ent_used.txt", "../../raw_data/passage2id/ent/ent_used")
    # convert_id("../../raw_data/query2id/ent/ent_query", "../../raw_data/entity/entity_map3.txt")
    # gen_test_data("../../raw_data/train/triples/")
    # count_used_entity("../../raw_data/entity/entity_map2.txt", "../../raw_data/entity/frequent_entity",
    #                  "../../raw_data/passage2id/ent/ent_passage", "../../raw_data/query2id/ent/ent_query")
    # with open("../../raw_data/train/triples/dev_ent_triples", 'w') as f_out:
    #     with open("../../raw_data/train/triples/ent_triples1") as f:
    #         for step, line in enumerate(f):
    #             if step > 999:
    #                 break
    #             f_out.write(line)
    # with open("../../raw_data/train/triples/dev_triples", "w") as f_out:
    #     with open("../../raw_data/train/triples/triples1") as f:
    #         for step, line in enumerate(f):
    #             if step > 999:
    #                 break
    #             f_out.write(line)
    # process_qrels("../../raw_data/top1000_dev/qrels.dev.small.tsv")
    gen_eval_data("../../raw_data/top1000_dev/word_dev", "../../raw_data/top1000_dev/ent_dev")

    args = parser.parse_args()


