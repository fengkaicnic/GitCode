__author__ = 'fengkai'
# coding=utf-8

import cv2
import os
from mtcnn import MTCNN
import sys

detector = MTCNN()

import pdb

# pdb.set_trace()
fname = 'D:/coded/Celeb-DF-v2/YouTube-real/'
# fname = 'Celeb-synthesis'
savepath = 'D:/coded/Celeb-DF-v2/youTube-real-cap/'
# if not os.path.exists(fname + '_vcon/'):
#     os.mkdir(fname + '_vcon/')


def save_img():
    videos = os.listdir(fname)
    ttt = 0
    # videos = ['1', '2', '3']
    for video_name in videos:
        # if video_name.split('_')[1] != 'id0' and ttt > 100:
        if ttt <= 160:
            ttt += 1
            continue
        ttt += 1
        file_name = video_name
        folder_name = video_name.split('.')[0]

        vc = cv2.VideoCapture(fname + '/' + file_name)  # 读入视频文件
        c = 1
        if vc.isOpened():  # 判断是否正常打开
            rval, frame = vc.read()
        else:
            rval = False

        timeF = 10  # 视频帧计数间隔频率
        total = 1
        # if not os.path.exists(savepath + folder_name):
        #     os.mkdir(savepath + folder_name)

        while rval and total:  # 循环读取视频帧
            try:
                rval, frame = vc.read()
                bb = detector.detect_faces(frame)
                if bb == []:
                    continue
                x, y, w, h = bb[0]['box']
                # realimgs.append(frame[y-10:y+h+10, x-10:x+w+10])
                if (c % timeF == 0):  # 每隔timeF帧进行存储操作
                    cv2.imwrite(savepath + folder_name+'_'+ str(c) + '.jpg',
                                frame[y - 10:y + h + 10, x - 10:x + w + 10])  # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
                    total = total - 1
                c = c + 1

                cv2.waitKey(1)
            except:
                continue
        vc.release()


save_img()
