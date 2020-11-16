import os
import cv2
import utils
import torch


# 读取记录过的人脸用于识别
# 返回格式 [[名称_0, 向量_0], [名称_1, 向量_1],...]
def loadLabel(net):
    labels = []
    path = './label'
    files = os.listdir(path)
    for file in files:
        label = []
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        faces = utils.faceDetection(utils.face_cascade, img)
        if len(faces) != 1:
            print(file, ': 人脸未检测到')
            continue
        face_img = utils.getFace(faces[0], img)
        face_img = utils.facePreProcess(face_img)
        y = torch.zeros(1, 1, 32, 32)
        y[0][0] = torch.tensor(face_img)
        y = y.to(utils.DEVICE)
        output = net(y)
        label.append(file)
        label.append(output)
        labels.append(label)
    return labels


def loadLabelPro():
    labels = []
    path = './label'
    files = os.listdir(path)
    for file in files:
        label = []
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        faces = utils.faceDetection(utils.face_cascade, img)
        if len(faces) != 1:
            print(file, ': 人脸未检测到')
            continue
        face_img = utils.getFace(faces[0], img)
        face_img = utils.facePreProcess2(face_img)
        output = utils.face_encoder.compute_face_descriptor(batch_img=[face_img])[0]
        label.append(file)
        label.append(output)
        labels.append(label)
    return labels
