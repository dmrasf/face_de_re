import cv2
import utils
import load_label
import predict
import train


# 绘制框及检测结果
def draw_face_result(img, net, labels, labels_pro, threshold=0.7):
    # 人脸检测
    faces = utils.faceDetection(utils.face_cascade, img)
    # 对每个人脸进行预测
    for (x, y, w, h) in faces:
        # 获取人脸图像并预处理
        face_img = utils.getFace([x, y, w, h], img)
        # 别人的
        who_pro = ''
        loss_pro = -1
        who = ''
        loss = -1
        who_pro, loss_pro = predict.predictionPro(face_img, labels_pro)
        # 矫正
        face_img_ail = utils.getAilgner(face_img)
        if face_img_ail is None:
            face_img = utils.facePreProcess(face_img)
        else:
            face_img = utils.facePreProcess(face_img_ail)
        # 自己的
        who, loss = predict.prediction(face_img, labels, net, threshold)
        # print('loss_pro', loss_pro)
        # 放置矩阵框和名称
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, who, (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX, w / 300, (255, 0, 255), 2)
        cv2.putText(img, who_pro, (x+int(w/2)+20, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, w / 300, (0, 0, 255), 2)
    return img


if __name__ == '__main__':
    # 加载网络 是否重新训练 参数
    net = train.getNet('params_200.pkl', False)
    is_pre = True
    if is_pre:
        # 加载已知人脸
        labels = load_label.loadLabel(net)
        labels_pro = load_label.loadLabelPro()
        # 参数为设备号
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, img = cap.read()
            # 预测及绘制信息
            img = draw_face_result(img, net, labels, labels_pro, 12)
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        # 验证准确率
        predict.testNet(net)
