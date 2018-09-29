#-*- coding:utf-8 -*-
#author:zhangwei

import os

imgs_path = '/home/zhangwei/videoaction/imgData/'
filelist = '/home/zhangwei/videoaction/filelist/'

with open(filelist + 'trainfilelist.txt' , 'w') as fw1:
    with open(filelist + 'testfilelist.txt' , 'w') as fw2:
        actionlist = os.listdir(imgs_path)
        ratio = 0.8
        label = 0
        for action in actionlist:
            actionimg_path = imgs_path + action
            videos = os.listdir(actionimg_path)
            videosum = len(videos)
            for i , video in enumerate(videos):
                if i < videosum * ratio:
                    fw1.write(action + '/' + video + ' ' + str(label) + '\n')
                else:
                    fw2.write(action + '/' + video + ' ' + str(label) + '\n')
            label += 1


