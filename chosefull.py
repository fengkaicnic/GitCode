#coding=utf-8

import cv2
import os
from mtcnn import MTCNN
import pdb
import math
import sys

#file_name = sys.argv[1]
path = '/mnt/Celeb-synthesis/'
detector = MTCNN()
save_path = '/mnt/celeb-synthesis-full/'
def save_img(path):
    tttmmm = 0
    for file_name in os.listdir(path):
        if file_name.split('_')[0] in ['id2', 'id1', 'id0']:
            continue
        tttmmm += 1
        if tttmmm % 5 == 0:
            print(file_name)
        nn = file_name.split('.')[0]
        vc = cv2.VideoCapture(path+file_name)
        c = 1
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
 
        timeF = 1 
        total = 5
 
        while rval and total:
            rval, frame = vc.read()
            try:
                bb = detector.detect_faces(frame)
            except:
                continue
            if bb == []:
                continue
            #pdb.set_trace()
            x,y,w,h = bb[0]['box']
            left_eye = bb[0]['keypoints']['left_eye'][1]
            right_eye = bb[0]['keypoints']['right_eye'][1]
            mmm = int(math.fabs(right_eye - left_eye))
            #realimgs.append(frame[y-10:y+h+10, x-10:x+w+10])
            if math.fabs(right_eye - left_eye) > 2:
                continue
            #pic_path = folder_name + '/'
            if (c % timeF == 0):
                try:
                    cv2.imwrite(save_path + nn + '_' + str(c) + '_' + str(mmm) + '.jpg', frame[y-20:y+h+20, x-20:x+w+20])
                    total = total -1
                except:
                    continue
            c = c + 1
            
            cv2.waitKey(1)
        vc.release()

save_img(path)

#[{'box': [181, 16, 60, 73], 'confidence': 0.9988462924957275, 'keypoints': {'left_eye': (197, 46), 'right_eye': (225, 40),
#'nose': (214, 55), 'mouth_left': (205, 74), 'mouth_right': (28, 70)}}]

