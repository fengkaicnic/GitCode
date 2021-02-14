import os
import pdb

class switch(object):
    def __init__(self, value):
    	self.value = value
    	self.fall = False

    def __iter__(self):
    	"""Return the match method once, then stop"""
    	yield self.match
    	raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

def getTestData(realpath, fakepath):
    #realpath = 'Celeb-real/'
    #fakepath = 'Celeb-synthesis/'

    dcttmp = {}
    realdct = {}
    realtdct = {}
    fakedct = {}
    faketdct = {}
    iddct = {}
    ids = ['id%d'% i for i in range(62)]
    for id in ids:
        iddct[id] = 0
        #pdb.set_trace()
    for name in os.listdir(realpath):
        ns = name.split('.')
        nms = ns[0].split('_')
        if iddct[nms[0]] < 9:
            realdct[ns[0]] = 1
            iddct[nms[0]] += 1
        elif iddct[nms[0]] < 10:
            realtdct[ns[0]] = 1
            iddct[nms[0]] += 1

    ids = ['id%d'% i for i in range(62)]
    for id in ids:
        iddct[id] = 0

    #pdb.set_trace()
    import numpy as np
    for name in os.listdir(fakepath):
        ns = name.split('.')
        nms = ns[0].split('_')
        if np.random.randint(5) > 1:
            continue
        if iddct[nms[0]] < 18:
            fakedct[ns[0]] = 1
            iddct[nms[0]] += 1
        elif iddct[nms[0]] < 20:
            faketdct[ns[0]] = 1
            iddct[nms[0]] += 1
    
    return realdct, realtdct, fakedct, faketdct
    #print(len(realdct), len(realtdct), len(faketdct), len(fakedct))

def translabel(labels):
	label = []
	for lab in labels:
		proba = max(lab)
		index = lab.index(proba)
		if index < 62:
			label.append(proba, 0)
		else:
			label.append(0, proba)

	return label
