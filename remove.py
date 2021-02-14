import os
import sys
import pdb

fdct = {}
path = sys.argv[1]
with open(path) as file:
    lines = file.readlines()
    for line in lines:
        fdct[line.strip().split(' ')[-1]] = 1

print(len(fdct))

path2 = sys.argv[2]
pdb.set_trace()
for name in os.listdir(path2):
    if name.split('.')[-1] != 'jpg':
        continue
    if fdct.get(name, None):
        continue
    os.remove(path2+name)

