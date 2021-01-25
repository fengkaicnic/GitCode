__author__ = 'fengkai'
# coding=utf-8

import cv2
import os
from mtcnn import MTCNN
import sys

detector = MTCNN()

import pdb

# pdb.set_trace()
fname = '/mnt/YouTube-real/'
# fname = 'Celeb-synthesis'
savepath = '/mnt/youtube-real-cap/'
# if not os.path.exists(fname + '_vcon/'):
#     os.mkdir(fname + '_vcon/')


def save_img():
    videos = os.listdir(fname)
    ttt = 0
    # videos = ['1', '2', '3']
    pdb.set_trace()
    for video_name in videos:
        # if video_name.split('_')[1] != 'id0' and ttt > 100:
        if ttt <= 112:
            ttt += 1
            continue
        ttt += 1
        file_name = video_name
        folder_name = video_name.split('.')[0]

        vc = cv2.VideoCapture(fname + '/' + file_name)
        c = 1
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False

        timeF = 10 
        total = 1
        # if not os.path.exists(savepath + folder_name):
        #     os.mkdir(savepath + folder_name)

        while rval and total:
            try:
                rval, frame = vc.read()
                bb = detector.detect_faces(frame)
                if bb == []:
                    continue
                x, y, w, h = bb[0]['box']
                # realimgs.append(frame[y-10:y+h+10, x-10:x+w+10])
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
