#-*- coding:utf-8 -*-
#author:zhangwei

import os

video_path = '/home/zhangwei/videoaction/videoData/'
videolist = os.listdir(video_path)

classeslist = '/home/zhangwei/videoaction/filelist/'
if not os.path.exists(classeslist):
    os.mkdir(classeslist)

with open(classeslist + 'classes.txt' , 'w') as fw:
    for i , video in enumerate(videolist):
        # print(type(video))
        fw.write(str(i) + ' ' + video + '\n')