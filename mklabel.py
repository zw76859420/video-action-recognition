#-*- coding:utf-8 -*-
#author:zhangwei

import os

imgs_path = '/home/zhangwei/videoaction/imgData/'
trainfilelist = '/home/zhangwei/videoaction/filelist/trainfilelist.txt'
testfilelist = '/home/zhangwei/videoaction/filelist/testfilelist.txt'

clip_length = 16

with open(trainfilelist , 'r') as fr1:
    with open('/home/zhangwei/videoaction/filelist/train.txt' , 'w') as fw1:
        trainlist = fr1.readlines()
        for line in trainlist:
            name = line.split(' ')[0]
            image_path = imgs_path + name
            label = line.split(' ')[-1]
            images = os.listdir(image_path)
            nb = len(images) // clip_length
            for i in range(nb):
                fw1.write(name + ' ' + str(i * clip_length + 1) + ' ' + label)

with open(testfilelist , 'r') as fr2:
    with open('/home/zhangwei/videoaction/filelist/test.txt' , 'w') as fw2:
        testlist = fr2.readlines()
        for line in testlist:
            name = line.split(' ')[0]
            image_path = imgs_path + name
            label = line.split(' ')[-1]
            images = os.listdir(image_path)
            nb = len(images) // clip_length
            for i in range(nb):
                fw2.write(name + ' ' + str(i * clip_length + 1) + ' ' + label)
