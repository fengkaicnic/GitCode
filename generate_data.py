import utils    
import numpy as np    
import copy    
import os
import pdb
import cv2
        
kernel = [[-1, 2, -2, 2, -1],    
           [2, -6, 8, -6, 2],    
           [-2, 8, -12, 8, -2],    
           [2, -6, 8, -6, 2],    
           [-1, 2, -2, 2, -1]]    
kernel = np.array((kernel), dtype="float32")    
filterr = 0        
def generate():    
    data = []    
    y = []    
    trainx = []    
    trainy = []    
    testx = []    
    testy = []    
    real_path = '/mnt/celeb-real-full/'    
    fake_path = '/mnt/celeb-synthesis-full/'    
    lstmnum = 4    
    ttname = ''    
    vdname = ''    
    fflag = 1    
    real_video_path = '/mnt/Celeb-real/'    
    fake_video_path = '/mnt/Celeb-synthesis/'    
        
    labelb = [0 for i in range(124)]    
    
    rdct, rtdct, fdct, ftdct = utils.getTestData(real_video_path, fake_video_path)    
    fnum = np.random.randint(1, 8)
    for name in os.listdir(real_path):    
        dd = []    
        #print(name)    
        vdname = '_'.join(name.split('_')[:2])    
        num = int(name.split('.')[0].split('_')[2])   
        #nb = '_' + name.split('.')[0].split('_')[-1] 
        nn = '_'.join(name.split('.')[0].split('_')[:2])    
        if num == fnum:    
            try:    
                for i in range(lstmnum):    
                    imgname = nn+'_'+str(num+i)+'.jpg'    
                    img = cv2.imread(real_path+imgname)    
                    img = cv2.resize(img, (128, 100))    
                    if filterr:    
                        img = cv2.filter2D(img, -1, kernel)    
                    dd.append(img)    
                dd = np.array(dd)    
            except:    
                print(name)    
                continue    
            nnum = name.split('_')[0].replace('id', '')    
            labellst = copy.deepcopy(labelb)    
            labellst[int(nnum)] = 1    
            #if total > alltotal * 0.2:    
            if rdct.get(vdname, None):     
                trainx.append(dd)    
                trainy.append(labellst)    
            else:    
                testx.append(dd)    
                testy.append(labellst)    
            ttname = vdname    
    #pdb.set_trace()    
    print(len(trainx), len(testx))    
    podata = len(testy)    
    fflag = 1    
        
    for name in os.listdir(fake_path):    
        #if np.random.randint(2) == 1:    
        #    continue    
        dd = []    
        vdname = '_'.join(name.split('_')[:3])   
        if not fdct.get(vdname, None):
            if not ftdct.get(vdname, None):
                continue
        num = int(name.split('.')[0].split('_')[3]) 
        # nb = '_' + name.split('.')[0].split('_')[-1]   
        nn = '_'.join(name.split('.')[0].split('_')[:3])    
        if num == 1:    
            try:    
                for i in range(lstmnum):    
                    imgname = nn + '_' + str(num + i) + '.jpg'    
                    img = cv2.imread(fake_path + imgname)    
                    img = cv2.resize(img, (128, 100))    
                    if filterr:    
                        img = cv2.filter2D(img, -1, kernel)    
                    dd.append(img)    
                dd = np.array(dd)    
            except:    
                print(name)    
                continue    
            nnum = name.split('_')[0].replace('id', '')    
            labellst = copy.deepcopy(labelb)    
            labellst[int(nnum)+62] = 1    
        
            if fdct.get(vdname, None):    
                trainx.append(dd)    
                trainy.append(labellst)    
            elif ftdct.get(vdname, None):    
                testx.append(dd)    
                testy.append(labellst)    
            ttname = vdname    
    negdata = len(testy) - podata    
    #pdb.set_trace()    
    import random
 
    seed = random.randint(0, 100)    
    random.seed(seed)    
    random.shuffle(trainx)    
    random.seed(seed)    
    random.shuffle(trainy)    
    random.seed(seed)    
    random.shuffle(testx)    
    random.seed(seed)    
    random.shuffle(testy)    
        
    #pdb.set_trace()    
    trainx = np.array(trainx)    
    trainy = np.array(trainy)    
    trainx = trainx.reshape(-1, lstmnum, 100, 128, 3)    
    trainx = trainx.astype('float32')    
    testx = np.array(testx)    
    testx = testx.reshape(-1, lstmnum, 100, 128, 3)    
    testx = testx.astype('float32')    
    testy = np.array(testy)    
    #pdb.set_trace()    
    print(len(trainx))    
    print(len(testx))    
    print(podata, negdata)    
    print(len(rdct), len(rtdct), len(fdct), len(ftdct))    
    return trainx, trainy, testx, testy    

 
