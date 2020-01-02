import numpy as np
import json
import argparse

# param --transe_emb: the path of binary embedding file
# param --transe_dim: the dimension of embedding. shall be 100

transe_vec = np.memmap('/home/student/raw_data/entity/entity2vec3.bin', dtype='float32', mode='r')
transe_vec = transe_vec.reshape(-1, 50)
print(transe_vec.shape)
print(transe_vec[0])

out = []

fst =  open('/home/student/raw_data/entity/entity_map3.txt', 'w')

line2 = 0

with open('/home/student/raw_data/entity/frequent_entity', 'r') as f:
    text_line = f.readlines()
    for line in text_line:
        ret = line.split('\t')
        aim = int(ret[2].strip())
        out.append(transe_vec[aim])
        fst.write(ret[0].strip()+'\t'+ret[2].strip()+'\t'+str(line2)+'\n')
        line2 = line2 + 1

fst.close()

f.close()
out = np.array(out)
print(out.shape)
out.tofile('/home/student/raw_data/entity/entity2vec4.bin')