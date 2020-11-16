import os
import numpy as np
import utils
import random
import cv2


# 格式化数据
def getData(a):
    data_0 = a[0]
    data_1 = a[1]
    data_size = len(data_0)
    datas = []
    # 自定义数据
    for i in range(data_size):
        # 第i张人脸图像
        anchor = data_0[i]
        # 同一个人的另一张图像
        positive = data_1[i]
        pos = i
        while pos == i:
            pos = random.randint(0, data_size-1)
        # 另一个人的图像
        negative = data_0[pos]
        data = [anchor, positive, negative]
        datas.append(data)
    return datas


# 获取训练数据
# 返回 nx3 的矩阵 n: 数据个数  3: 三种脸部图像
def getTrainData(data_name):
    if not os.path.exists(data_name):
        print('获取训练数据...')
        loadData(data_name, './trainset')
    a = np.load(data_name)
    # 返回格式化后的数据
    return getData(a)


# 返回 nx3 的矩阵 n: 数据个数  3: 三种脸部图像
def getTestData(data_name):
    if not os.path.exists(data_name):
        print('获取测试数据...')
        loadData(data_name, './testset')
    a = np.load(data_name)
    return getData(a)


# 从文件夹读取人脸图像
# 包括人脸检测、矫正、归一化，保存为np.array
def loadData(data_name, path):
    data_1 = []
    data_2 = []
    files = os.listdir(path)
    count = 1
    while count < len(files)/2:
        faces_img = [[], []]
        for i in range(2):
            file_name = str(count) + '_' + str(i) + '.jpg'
            file_path = os.path.join(path, file_name)
            # 读取图片
            img = cv2.imread(file_path)
            faces = utils.faceDetection(utils.face_cascade, img)
            if len(faces) == 0:
                break
            elif len(faces) > 1:
                # 若有多个人脸，那么选择靠近中间的
                m = 250
                face_pos = 0
                for pos, [x, y, w, h] in enumerate(faces):
                    drift = abs(x+w/2-125)+abs(y+h/2-125)
                    if drift < m:
                        face_pos = pos
                face_x, face_y, face_w, face_h = faces[face_pos]
            else:
                face_x, face_y, face_w, face_h = faces[0]
            face_img = utils.getFace([face_x, face_y, face_w, face_h], img)
            # 图像矫正
            face_img = utils.getAilgner(face_img)
            if face_img is None:
                break
            face_img = utils.facePreProcess(face_img)
            faces_img[i] = face_img
        if len(faces_img[0]) and len(faces_img[1]):
            data_1.append(faces_img[0])
            data_2.append(faces_img[1])
        count += 1
    data = [data_1, data_2]
    a = np.array(data)
    np.save(data_name, a)
    print('数据已保存为: ', data_name)

# loadData('test', 'trainset')
