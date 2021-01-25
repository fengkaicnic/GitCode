import os
import shutil
import numpy as np

fpath = '/mnt/celeb-synthesis-lstm/'
tpath = '/mnt/celeb-synthesis-rand-lstm/'
for fname in os.listdir(fpath):
    if np.random.randint(7) < 3:
        shutil.move(fpath+fname, tpath)

