from torch import nn
import torch
import torch.nn.functional as F
import os
import load_data
import utils


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        # 32x32 -> 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 14x14 -> 10x10 -> 5x5
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def trainModel(save_name):
    # 训练数据
    data_name = 'facedata.npy'
    net = Net().to(utils.DEVICE)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    # margin: positive 与 negative 间最少距离
    criterion = nn.TripletMarginLoss(margin=2)
    params = save_name.split('_')
    params = params[1].split('.')
    epochs = int(params[0])
    for epoch in range(epochs):
        dataset = load_data.getTrainData(data_name)
        print('第{}次训练'.format(epoch+1))
        # batch = 1
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
            loss = criterion(output_0, output_1, output_2)
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
    # 保存参数
    print('保存模型...')
    torch.save(net.state_dict(), save_name)
    return net


def getNet(param_name, is_re_train=False):
    net = Net().to(utils.DEVICE)
    if not os.path.exists(param_name) or is_re_train:
        print('训练模型...')
        trainModel(param_name)
    print('加载模型...')
    net.load_state_dict(torch.load(param_name))
    return net
