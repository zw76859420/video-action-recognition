#-*- ahthor:zhangwei -*-
#author:zhangwei

import cv2
import os

video_path = '/home/zhangwei/videoaction/videoData/'
save_path = '/home/zhangwei/videoaction/imgData/'

actionlist = os.listdir(video_path)
# print(len(actionlist))

for action in actionlist:
    if not os.path.exists(save_path + action):
        os.makedirs(save_path + action)

    # print(action)

    videolist = os.listdir(video_path + action)
    # print(videolist)


    for video in videolist:
        prefix = video.split('.')[0]
        if not os.path.exists(save_path + action + '/' + prefix):
            os.mkdir(save_path + action + '/' + prefix)

        save_name = save_path + action + '/' + prefix + '/'
        video_name = video_path + action + '/' + video

        cap = cv2.VideoCapture(video_name)
        # print(cap)
        fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #print(fps)

        fps_count = 0

        for i in range(fps):
            ret , frame = cap.read()
            if ret:
                cv2.imwrite(save_name + str(10000 + fps_count) + '.jpg' , frame)
                fps_count += 1
