import cv2
import numpy as np
import torch
import openface
import dlib
import face_recognition_models

# 训练好的网络
face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# 配置参数 人脸检测参数
face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'./haarcascade_eye.xml')
eye2_cascade = cv2.CascadeClassifier(r'./haarcascade_eye_tree_eyeglasses.xml')
predictor_model = "shape_predictor_68_face_landmarks.dat"
face_aligner = openface.AlignDlib(predictor_model)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
def facePreProcess(face):
    # 灰度化
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # 缩放至32x32
    face = cv2.resize(face, (32, 32))
    # 归一化
    face = np.array(face)
    face = (face / 255.).astype(np.float32)
    return face


# 图像预处理
def facePreProcess2(face):
    face = cv2.resize(face, (150, 150))
    # # 归一化
    # face = np.array(face)
    # face = (face / 255.).astype(np.float32)
    return face


# 检测脸部一副图像可能有多个人脸
def faceDetection(face_cascade, img):
    # scaleFactor: 尺寸缩放  minNeighbors: 越大越精确去掉误识别的脸 minSize: 小于将会忽略
    faces = face_cascade.detectMultiScale(
        img, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces


# 人眼检测，用于图像矫正
def eyeDetection(face_cascade, img):
    faces = face_cascade.detectMultiScale(
        img, scaleFactor=1.1, minNeighbors=6, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces


# 获取脸部图像灰度 pos: [x, y, w, h] 左上角及人脸的宽高
def getFace(pos, img):
    x = pos[0]
    y = pos[1]
    w = pos[2]
    h = pos[3]
    # 获取脸部图像
    face = img[y:h + y, x:h + x]
    return face


# 32x32 -> (1, 1, 32, 32)
def listToTensor(data):
    a = torch.rand(1, 1, 32, 32)
    a[0][0] = torch.tensor(data)
    return a


# 32x32 -> (1, 1, 32, 32)
def listToTensor2(face_img):
    a = torch.rand(1, 3, 96, 96)
    a[0][0] = torch.tensor(face_img[:, :, 0])
    a[0][1] = torch.tensor(face_img[:, :, 1])
    a[0][2] = torch.tensor(face_img[:, :, 2])
    return a


def getAilgner(face_img):
    return face_aligner.align(534, face_img)
