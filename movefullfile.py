import os
import shutil

tpath = '/mnt/celeb-real-full/'
fpath = '/mnt/celeb-synthesis-full/'


for name in os.listdir(fpath):
    vst = '_'.join(name.split('.')[0].split('_')[:-1])
    shutil.move(fpath + name, fpath + vst + '.jpg')

