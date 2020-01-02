import tagme
import sys
import logging
import json
import time
import argparse
import numpy as np
import os
import multiprocessing
import re

tagme.GCUBE_TOKEN = 'cad23c26-6f1f-4164-a62e-fc5107c031ab-843339462'
Entities = dict()
exist_line = 0
starttime = time.time()
total_size = 0
processed = 0
percentage = 0


def get_entities(ent_path):
    with open(ent_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            Entities[tokens[0]] = int(tokens[1])


def tag_text(text):
    while True:
        try:
            annotations = tagme.annotate(text)
            break
        except:
            (text + "Again!")
            pass
    entities_list = set()
    # Print annotations with a score higher than 0.1
    if annotations:
        for ann in annotations.get_annotations(0.1):
            # begin = int(ann.begin)
            # end = int(ann.end)
            # score = float(ann.score)
            entity_title = str(ann.entity_title)
            # if entity_title not in Entities:
            #     print("missing: " + entity_title)
            # entity_title = entity_title.strip().replace(" ", "_").replace(";", "-COLON-").replace("(", "-LRB-").replace(")", "-RRB-")
            entity_title = re.sub("\s+", " ", entity_title)
            # entities_list.append([begin, end, entity_title, float(score)])

            entities_list.add(entity_title)
    entities_list = list(entities_list)
    return entities_list


def read_file(path):
    datas = list()
    global exist_line
    global total_size
    global processed
    global percentage
    total_size = os.path.getsize(path)

    with open(path, 'r') as f:
        for step, line in enumerate(f):
            if step < exist_line:
                processed += len(line)
                percentage = (int) (processed / total_size)
                continue
            # data = json.loads(line)
            # if data["doc_id"] not in exist_data:
            #     datas.append(data)
            datas.append(line)
    return datas


def read_exist_file(path):
    if not os.path.exists(path):
        out = open(path, "w")
        out.close()
    datas = dict()
    global exist_line
    with open(path) as f:
        for step, line in enumerate(f):
            # data = json.loads(line)
            # datas[data["doc_id"]] = 1
            exist_line += 1
    return datas


def process_file(datas, path, num, results):
    with open(path, "w") as f:
        global processed
        global percentage
        for counter, data in enumerate(datas):
            # evidence_map = data["predicted_evidence"]
            # evidence_sent = data["evidence"]
            # assert len(evidence_map) == len(evidence_sent)
            # evidence = list()
            # for step in range(len(evidence_map)):
            #    sent = evidence_sent[step][0]
            #    sent = " ".join(sent.strip().split())
            #    if len(sent) != 0:
            #        entities = tag_text(sent)
            #        evidence.append([evidence_map[step][0], evidence_map[step][1], evidence_map[step][2], sent, entities])
            # try:
            #   del data["predicted_evidence"]
            # except:
            #    pass
            processed += len(data)
            if int(processed / total_size) > percentage:
                percentage = int(processed / total_size)
                print(str(percentage) + "% processed")
                print("time used: %d s" % (time.time() - starttime))
            tokens = data.strip().split('\t')
            if len(tokens) != 2:
                continue
            entities = tag_text(tokens[1])
            # for t in tokens:
            #     ent = tag_text(t)
            #     entities.append(ent)
            object = list()
            for e in entities:
                if e in Entities:
                    object.append(Entities[e])

            # object = [[], [], []]
            # for i, ent in enumerate(entities):
            #     for e in ent:
            #         if e in Entities:
            #             object[i].append(Entities[e])
            # obj = {'query': object[0], 'pos_passage': object[1], 'neg_passage': object[2]}
            obj = {'id': tokens[0], 'passage': object}
            obj = json.dumps(obj)
            f.write(obj)
            f.write('\n')
            results[num].append(obj)
            print(counter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ent_path')
    parser.add_argument('--thread', required=True, type=int)
    args = parser.parse_args()
    starttime = time.time()
    print("Read file" + args.input)
    get_entities(args.ent_path)
    # exist_data = read_exist_file(args.output)
    datas = read_file(args.input)
    print("Start linking")
    # below is for multi-thread
    results = [[] for i in range(args.thread)]
    pool = multiprocessing.Pool(processes=args.thread)
    result = list()
    window = int(np.ceil(len(datas) * 1.0 / args.thread))
    for i in range(0, args.thread):
        result.append(pool.apply_async(process_file, (datas[i * window: (i + 1) * window], args.output + str(i), i, results,)))
    pool.close()
    pool.join()

    # process_file(datas, args.output)
    print("time used: %d" % (time.time() - starttime))
    print("Finish")


