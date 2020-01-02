import tagme
import sys
import logging
import json
import time
import argparse
import numpy as np
import os
import multiprocessing

tagme.GCUBE_TOKEN = 'cad23c26-6f1f-4164-a62e-fc5107c031ab-843339462'



def tag_text(text):
    while (1):
        try:
            annotations = tagme.annotate(text)
            break
        except:
            print(text + "Again!")
            pass
    entities_list = list()
    # Print annotations with a score higher than 0.1
    if annotations:
        for ann in annotations.get_annotations(0.1):
            begin = int(ann.begin)
            end = int(ann.end)
            score = float(ann.score)
            entity_title = str(ann.entity_title)
            entity_title = entity_title.strip().replace(" ", "_").replace(";", "-COLON-").replace("(", "-LRB-").replace(")", "-RRB-")
            entities_list.append([begin, end, entity_title, float(score)])
    return entities_list



def read_file(exist_data, path):
    datas = list()
    with open(path) as f:
        for step, line in enumerate(f):
            data = json.loads(line)
            if data["id"] not in exist_data:
                datas.append(data)
    return datas

def read_exist_file(path):
    if not os.path.exists(path):
        out = open(path, "w")
        out.close()
    datas = dict()
    with open(path) as f:
        for step, line in enumerate(f):
            data = json.loads(line)
            datas[data["id"]] = 1
    return datas




def process_file(datas, path):
    with open(path, "a+") as f:
        for counter, data in enumerate(datas):
            #evidence_map = data["predicted_evidence"]
            #evidence_sent = data["evidence"]
            #assert len(evidence_map) == len(evidence_sent)
            #evidence = list()
            #for step in range(len(evidence_map)):
            #    sent = evidence_sent[step][0]
            #    sent = " ".join(sent.strip().split())
            #    if len(sent) != 0:
            #        entities = tag_text(sent)
            #        evidence.append([evidence_map[step][0], evidence_map[step][1], evidence_map[step][2], sent, entities])
            #try:
            #   del data["predicted_evidence"]
            #except:
            #    pass
            entities = tag_text(data["claim"])
            data["claim_ent"] = entities
            #data["evidence"] = evidence
            line = json.dumps(data)
            f.write(line + "\n")
            print (counter)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--thread', required=True, type=int)
    args = parser.parse_args()
    print ("Read file" + args.input)
    exist_data = read_exist_file(args.output)
    datas = read_file(exist_data, args.input)
    print ("Start linking")
    pool = multiprocessing.Pool(processes=args.thread)
    result = list()
    window = int(np.ceil(len(datas) * 1.0 / args.thread))
    for i in range(0, args.thread):
        result.append(pool.apply_async(process_file, (datas[i * window: (i + 1) * window], args.output, )))
    pool.close()
    pool.join()
    print ("Finish")