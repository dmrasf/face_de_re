import torch
import train
import numpy as np
import torch.nn.functional as F
import load_data
import utils


# 使用label里的数据与未知脸匹配 返回名称和偏差
# face: 处理好的人脸图像
def prediction(face, labels, net, threshold=0.7):
    if len(labels) == 0:
        return 'unknown', -1
    x = torch.zeros(1, 1, 32, 32)
    x[0][0] = torch.tensor(face)
    x = x.to(utils.DEVICE)
    # 求特征向量
    output = net(x)
    euclidean_distance = 10000
    who = 'unknown'
    for i, label in enumerate(labels):
        # 求两个向量的距离
        e = F.pairwise_distance(label[1], output, keepdim=True)
        # 寻找距离最近的符合的人脸
        if euclidean_distance > e:
            euclidean_distance = e
            who = label[0]
    # 最相似人脸若大于阈值则该脸为未知脸
    if euclidean_distance > threshold:
        who = 'unknown'
    return who, euclidean_distance


def predictionPro(face, labels_pro, threshold=0.6):
    if len(labels_pro) == 0:
        return 'unknown', -1
    encoding = utils.face_encoder.compute_face_descriptor(batch_img=[utils.facePreProcess2(face)])[0]
    euclidean_distance = 10000
    who = 'unknown'
    for i, label in enumerate(labels_pro):
        # 求两个向量的距离
        e = np.linalg.norm(np.array(label[1]) - np.array(encoding))
        # 寻找距离最近的符合的人脸
        if euclidean_distance > e:
            euclidean_distance = e
            who = label[0]
    # 最相似人脸若大于阈值则该脸为未知脸
    if euclidean_distance > threshold:
        who = 'unknown'
    return who, euclidean_distance


# 测试网络 返回准确率
def testNet(net, thMin=0.2, thMax=15):
    # dataset = load_data.getTestData('facedata.npy')
    dataset = load_data.getTestData('test_data.npy')
    count = 0
    ths = []
    rights = []
    th = thMin
    while th < thMax:
        th += 0.1
        ths.append(th)
        rights.append(0)
    for i, [anchor, positive, negative] in enumerate(dataset):
        anchor = utils.listToTensor(anchor)
        positive = utils.listToTensor(positive)
        negative = utils.listToTensor(negative)
        anchor = anchor.to(utils.DEVICE)
        positive = positive.to(utils.DEVICE)
        negative = negative.to(utils.DEVICE)
        output_0 = net(anchor)
        output_1 = net(positive)
        output_2 = net(negative)
        # 距离
        e_pos = F.pairwise_distance(output_0, output_1, keepdim=True)
        e_neg = F.pairwise_distance(output_0, output_2, keepdim=True)
        # 计算合适的阈值
        for j in range(len(ths)):
            threshold = ths[j]
            if e_pos < threshold:
                rights[j] += 1
            if e_neg > threshold:
                rights[j] += 1
        count += 1
    for i in range(len(ths)):
        threshold = ths[i]
        print(threshold, ': ', rights[i], count * 2, rights[i] / count / 2)
