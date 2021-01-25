# coding=utf-8

import cv2
import os
from mtcnn import MTCNN
import sys
import numpy as np

detector = MTCNN()

import pdb

# pdb.set_trace()
fname = '/mnt/Celeb-synthesis/'
# fname = 'Celeb-synthesis'
savepath = '/mnt/celeb-synthesis-eye30/'
# if not os.path.exists(fname + '_vcon/'):
#     os.mkdir(fname + '_vcon/')

vname = np.load('vname.npy')
vname = vname.item()
def save_img():
    videos = os.listdir(fname)
    ttt = 0
    # videos = ['1', '2', '3']
    pdb.set_trace()
    for video_name in videos:
        nss = video_name.split('.')
        if vname.get(nss[0], None):
            continue
        ttt += 1
        if ttt%10 == 0:
            print(video_name)
        # if video_name.split('_')[1] != 'id0' and ttt > 100:
        file_name = video_name
        folder_name = video_name.split('.')[0]

        vc = cv2.VideoCapture(fname + '/' + file_name)
        c = 1
        if vc.isOpened(): 
            rval, frame = vc.read()
        else:
            rval = False

        timeF = 1
        total = 7
        # if not os.path.exists(savepath + folder_name):
        #     os.mkdir(savepath + folder_name)

        while rval and total:
            try:
                rval, frame = vc.read()
                bb = detector.detect_faces(frame)
                if bb == []:
                    continue
                x, y, w, h = bb[0]['box']
                lefteye = bb[0]['keypoints']['left_eye']
                righteye = bb[0]['keypoints']['right_eye']
                if lefteye[0] - x > 30 and x + w - righteye[0] > 30:
                    #print(lefteye[0] - x, x + w -righteye[0])
                    continue
                print(lefteye[0] - x, x + w -righteye[0])
                # realimgs.append(frame[y-10:y+h+10, x-10:x+w+10])
                pic_path = folder_name + '/'
                if (c % timeF == 0):
                    cv2.imwrite(savepath + folder_name+'_'+ str(c) + '.jpg',
                                frame[y - 10:y + h + 10, x - 10:x + w + 10])
                    total = total - 1
                c = c + 1

                cv2.waitKey(1)
            except:
                continue
        vc.release()


save_img()

