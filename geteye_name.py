import os
import pdb
import numpy as np

path = '/mnt/celeb-synthesis-eye/'

vname = {}
pdb.set_trace()
for name in os.listdir(path):
    ns = name.split('_')
    #if vname.has_key('_'.join(ns[:3])):
    if vname.get('_'.join(ns[:3]), None):
        continue
    vname['_'.join(ns[:3])] = 1

for key in vname.keys():
    print(key)

print(len(vname))
np.save('vname', vname)

