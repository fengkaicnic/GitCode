import os
import shutil

fpath = ''
mpath = ''
fdct = {}
filepath = ''
with open('propert.txt', 'r') as file:
	lines = file.readlines()
	for line in lines:
		lst = line.strip().split('\t')
		if lst[1] == '0':
			fdct[lst[0]] = fpath
		else:
			fdct[lst[0]] = mpath

for videoname in os.listdir(filepath):
	vst = videoname.split('_')[0]
	shutil.move(filepath+videoname, fdct[vst]+videoname)
