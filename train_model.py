#-*- coding:utf-8 -*-
#author:zhangwei

from videoaction_recognition.model import c3d_model
from keras.optimizers import SGD , Adam , RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import numpy as np
import random
import cv2
import os

def preprocess(inputs):
    inputs /= 255.
    inputs -= 0.5
    inputs *= 2.
    return inputs

def preprocess_batch(lines , img_path , train=True):
    num = len(lines)
    # print(num)
    batch = np.zeros(shape=[num , 16 , 90 , 120 , 3] , dtype='float32')
    labels = np.zeros(shape=[num] , dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]                               #读取文件路径位置；
        label = lines[i].split(' ')[-1]                             #读取文件对应的标签位置；
        symbol = lines[i].split(' ')[1]                             #读取帧数间隔的帧的标签位置；
        label = label.strip('\n')                                   #去掉标签最后的换行符号；
        label = int(label)                                          #对标签进行整数化处理；
        symbol = int(symbol) - 1
        imgs = os.listdir(img_path + path)                          #遍历文件夹下图片的位置；
        imgs.sort(key=str.lower)                                    #由于listdir读取数据是随机排序，这时候需要对遍历的文件夹进行排序，sort对图像进行排序；
        for j in range(16):
            img = imgs[symbol + j]  # 读取图像对应的16帧数据；
            image = cv2.imread(img_path + path + '/' + img)         # 分别读取图像；
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # 对图像的通道进行转换；
            image = cv2.resize(image, (120, 90))                    # 对图像的大小进行规整为【120 ， 90】；
            # print(image)
            batch[i][j][:][:][:] = image
            # print(batch)
        labels[i] = label
    # print(labels[1000])
    return batch , labels

def generator_train_batch(train_txt , batch_size , num_classes , img_path):
    with open(train_txt , 'r') as fr:
        lines = fr.readlines()
        num = len(lines)
        while True:
            new_line = []
            index = [n for n in range(num)]
            random.shuffle(index)
            # print(index)
            for m in range(num):
                new_line.append(lines[index[m]])
            # print(len(new_line))
            for i in range(int(num / batch_size)):
                a = i * batch_size
                b = (i + 1) * batch_size
                x_train , x_labels = preprocess_batch(new_line[a : b] , img_path , train=True)
                x = preprocess(x_train)
                y = np_utils.to_categorical(np.array(x_labels) , num_classes)
                x = np.transpose(x , (0,2,3,1,4))
                yield x , y

def generator_test_batch(test_txt , batch_size , num_classes , img_path):
    with open(test_txt , 'r') as fr:
        lines = fr.readlines()
        num = len(lines)
        while True:
            new_line = []
            index = [n for n in range(num)]
            random.shuffle(index)
            # print(index)
            for m in range(num):
                new_line.append(lines[index[m]])
            # print(len(new_line))
            for i in range(int(num / batch_size)):
                a = i * batch_size
                b = (i + 1) * batch_size
                x_train , x_labels = preprocess_batch(new_line[a : b] , img_path , train=True)
                x = preprocess(x_train)
                y = np_utils.to_categorical(np.array(x_labels) , num_classes)
                x = np.transpose(x , (0,2,3,1,4))
                yield x , y


if __name__ == '__main__':
    #设置最大显存不超过90%；
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    set_session(tf.Session(config=config))

    imgs_path = '/home/zhangwei/videoaction/imgData/'
    train_file = '/home/zhangwei/videoaction/filelist/train.txt'
    test_file = '/home/zhangwei/videoaction/filelist/test.txt'

    with open(train_file , 'r') as fr1:
        lines = fr1.readlines()
        # print(lines)
        train_samples = len(lines)
        op = Adam(lr=0.1 , beta_1=0.9 , beta_2=0.999)
        op1 = RMSprop(lr=0.01)
        model = c3d_model()
        model.compile(loss='categorical_crossentropy' , optimizer=op , metrics=['accuracy'])
        model.fit_generator(generator_train_batch(train_txt=train_file , batch_size=1 , num_classes=51 , img_path=imgs_path) ,
                            steps_per_epoch=train_samples // 1 ,
                            epochs=10)

        # a = generator_train_batch(train_txt=train_file , batch_size=128 , num_classes=51 , img_path=imgs_path)
        # for i in a:
        #     print(i[0].shape , i[1].shape)

    # with open(test_file , 'r') as fr2:
    #     lines2 = fr2.readlines()
    #     b = generator_test_batch(test_file , batch_size=128 , num_classes=51 , img_path=imgs_path)
    #     for i in b:
    #         print(i[0].shape , i[1].shape)