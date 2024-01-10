import torch.nn as nn
import torch
from skimage import color
'''

    这里做了一个双通道的卷积神经网络，两个通道分别传入x1,x2两个张量进行运算，输入张量尺寸为32*32，
    张量在经过两次卷积池化再拉直后尺寸变为1（3255），两个张量进行拼接进入Linear层尺寸变为2*（3255）。

'''


class C_lenet(nn.Module):
    def __init__(self, nn_number, class_number, lstm_layer):
        super(C_lenet, self).__init__()
        self.nn_classnumber = class_number
        self.nn_lstm_layer = lstm_layer
        self.nn_number = nn_number
        self.conv1_1 = nn.Sequential(
            # 默认步长为1
            nn.Conv2d(3, 8, 5),  # 输入通道数为3，输出通道数为16.
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # # CBAM添加到第一层之前
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(8, 16, 5),  # 使用32个大小为5*5的卷积核
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(3, 8, 5),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(8, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2_4 = nn.Sequential(
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.conv_3_to_1 = nn.Sequential(
            nn.Conv2d(3, 1, 1),
        )
        self.Maxpool_2d = nn.AvgPool2d(2, 2)
        self.adaptMaxpool_2d = nn.AdaptiveAvgPool2d((4, 4))
        self.lstm = nn.LSTM(input_size=2 * 32 * 4 * 4,
                            hidden_size=128, num_layers=2,
                            batch_first=True)
        self.lstm_parallel_1 = nn.LSTM(input_size=61,
                            hidden_size=122, num_layers=1,
                            batch_first=True)
        self.lstm_parallel_2 = nn.LSTM(input_size=61,
                                       hidden_size=122, num_layers=2,
                                       batch_first=True)
        self.lstm_parallel_3 = nn.LSTM(input_size=61,
                                       hidden_size=122, num_layers=3,
                                       batch_first=True)
        # 如果还不满意，可以改4*4这个参数
        self.fc1_pCRNN = nn.Linear(2 * 32 * 4 * 4 + 2 * 122, 2 * 1024)
        self.fc1_pCNN = nn.Linear(2 * 32 * 4 * 4, 2 * 1024)
        self.fc1_CRNN = nn.Linear(32 * 4 * 4 + 122, 2 * 512)
        self.fc1_CNN = nn.Linear(32 * 4 * 4, 2 * 512)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(p=0.5)
        # 最后一层fc3数目表示最终分类的类别。
        self.fc3_2 = nn.Linear(2 * 1024, self.nn_classnumber)
        self.fc3 = nn.Linear(2 * 512, self.nn_classnumber)

    def forward(self, x1, x2):  # 按照1，2，3，4的顺序全部进行训练？
        # nn_number=1, 则为pcrnn, 2则为pcnn, 3则为crnn-only doa, 4则为cnn-only doa

        if self.nn_number == 1:
            if self.nn_lstm_layer == 1:  # pcrnn
                x1_1 = self.conv1_1(x1)
                x1_1 = self.Maxpool_2d(x1_1)
                x1_1 = self.conv1_2(x1_1)
                x1_1 = self.conv1_3(x1_1)
                # print(x1_1.size())
                # x1_1 = self.ca(x1_1) * x1_1
                # x1_1 = self.sa(x1_1) * x1_1

                x1_1 = self.adaptMaxpool_2d(x1_1)

                x1_1 = torch.flatten(x1_1, 1)
                # 将图像转为灰度图
                x1_2 = x1.cpu()
                x1_2 = color.rgb2gray(x1_2)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x1_2 = torch.from_numpy(x1_2).to(device)
                x1_2, _ = self.lstm_parallel_1(x1_2)
                x1_sum = torch.cat((x1_1, x1_2[:, -1, :]), 1)

                x2_1 = self.conv2_1(x2)

                x2_1 = self.Maxpool_2d(x2_1)
                x2_1 = self.conv2_2(x2_1)
                x2_1 = self.conv2_3(x2_1)

                ## x2_1 = self.ca(x2_1) * x2_1
                ## x2_1 = self.sa(x2_1) * x2_1

                x2_1 = self.adaptMaxpool_2d(x2_1)
                x2_1 = torch.flatten(x2_1, 1)
                ## 通过卷积将三通道图像转换为单通道图像。
                x2_2 = x2.cpu()
                x2_2 = color.rgb2gray(x2_2)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x2_2 = torch.from_numpy(x2_2).to(device)
                x2_2, _ = self.lstm_parallel_1(x2_2)
                x2_sum = torch.cat((x2_1, x2_2[:, -1, :]), 1)
                x = torch.cat((x1_sum, x2_sum), 1)
                x = self.fc1_pCRNN(x)
                # x = self.fc2(x)
                x = self.ReLU(x)
                x = self.dropout(x)
                x = self.fc3_2(x)
                # softmax被集成到交叉熵损失里面，所以在网络最后不需要添加softmax.
                return x
            elif self.nn_lstm_layer == 2:  # pcrnn
                x1_1 = self.conv1_1(x1)
                x1_1 = self.Maxpool_2d(x1_1)
                x1_1 = self.conv1_2(x1_1)
                x1_1 = self.conv1_3(x1_1)
                # print(x1_1.size())
                # x1_1 = self.ca(x1_1) * x1_1
                # x1_1 = self.sa(x1_1) * x1_1

                x1_1 = self.adaptMaxpool_2d(x1_1)

                x1_1 = torch.flatten(x1_1, 1)
                # 将图像转为灰度图
                x1_2 = x1.cpu()
                x1_2 = color.rgb2gray(x1_2)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x1_2 = torch.from_numpy(x1_2).to(device)
                x1_2, _ = self.lstm_parallel_2(x1_2)
                x1_sum = torch.cat((x1_1, x1_2[:, -1, :]), 1)

                x2_1 = self.conv2_1(x2)

                x2_1 = self.Maxpool_2d(x2_1)
                x2_1 = self.conv2_2(x2_1)
                x2_1 = self.conv2_3(x2_1)

                ## x2_1 = self.ca(x2_1) * x2_1
                ## x2_1 = self.sa(x2_1) * x2_1

                x2_1 = self.adaptMaxpool_2d(x2_1)
                x2_1 = torch.flatten(x2_1, 1)
                ## 通过卷积将三通道图像转换为单通道图像。
                x2_2 = x2.cpu()
                x2_2 = color.rgb2gray(x2_2)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x2_2 = torch.from_numpy(x2_2).to(device)
                x2_2, _ = self.lstm_parallel_2(x2_2)
                x2_sum = torch.cat((x2_1, x2_2[:, -1, :]), 1)
                x = torch.cat((x1_sum, x2_sum), 1)
                x = self.fc1_pCRNN(x)
                # x = self.fc2(x)
                x = self.ReLU(x)
                x = self.dropout(x)
                x = self.fc3_2(x)
                # softmax被集成到交叉熵损失里面，所以在网络最后不需要添加softmax.
                return x
            elif self.nn_lstm_layer == 3:  # pcrnn
                x1_1 = self.conv1_1(x1)
                x1_1 = self.Maxpool_2d(x1_1)
                x1_1 = self.conv1_2(x1_1)
                x1_1 = self.conv1_3(x1_1)
                # print(x1_1.size())
                # x1_1 = self.ca(x1_1) * x1_1
                # x1_1 = self.sa(x1_1) * x1_1

                x1_1 = self.adaptMaxpool_2d(x1_1)

                x1_1 = torch.flatten(x1_1, 1)
                # 将图像转为灰度图
                x1_2 = x1.cpu()
                x1_2 = color.rgb2gray(x1_2)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x1_2 = torch.from_numpy(x1_2).to(device)
                x1_2, _ = self.lstm_parallel_3(x1_2)
                x1_sum = torch.cat((x1_1, x1_2[:, -1, :]), 1)

                x2_1 = self.conv2_1(x2)

                x2_1 = self.Maxpool_2d(x2_1)
                x2_1 = self.conv2_2(x2_1)
                x2_1 = self.conv2_3(x2_1)

                ## x2_1 = self.ca(x2_1) * x2_1
                ## x2_1 = self.sa(x2_1) * x2_1

                x2_1 = self.adaptMaxpool_2d(x2_1)
                x2_1 = torch.flatten(x2_1, 1)
                ## 通过卷积将三通道图像转换为单通道图像。
                x2_2 = x2.cpu()
                x2_2 = color.rgb2gray(x2_2)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x2_2 = torch.from_numpy(x2_2).to(device)
                x2_2, _ = self.lstm_parallel_3(x2_2)
                x2_sum = torch.cat((x2_1, x2_2[:, -1, :]), 1)
                x = torch.cat((x1_sum, x2_sum), 1)
                x = self.fc1_pCRNN(x)
                # x = self.fc2(x)
                x = self.ReLU(x)
                x = self.dropout(x)
                x = self.fc3_2(x)
                # softmax被集成到交叉熵损失里面，所以在网络最后不需要添加softmax.
                return x
        if self.nn_number == 2:  # pcnn
            x1_1 = self.conv1_1(x1)
            x1_1 = self.Maxpool_2d(x1_1)
            x1_1 = self.conv1_2(x1_1)
            x1_1 = self.conv1_3(x1_1)
            # print(x1_1.size())
            # x1_1 = self.ca(x1_1) * x1_1
            # x1_1 = self.sa(x1_1) * x1_1
            x1_1 = self.adaptMaxpool_2d(x1_1)

            x1_1 = torch.flatten(x1_1, 1)

            x2_1 = self.conv2_1(x2)
            x2_1 = self.Maxpool_2d(x2_1)
            x2_1 = self.conv2_2(x2_1)
            x2_1 = self.conv2_3(x2_1)

            ## x2_1 = self.ca(x2_1) * x2_1
            ## x2_1 = self.sa(x2_1) * x2_1

            x2_1 = self.adaptMaxpool_2d(x2_1)
            x2_1 = torch.flatten(x2_1, 1)
            x = torch.cat((x1_1, x2_1), 1)
            x = self.fc1_pCNN(x)
            # x = self.fc2(x)
            x = self.ReLU(x)
            x = self.dropout(x)
            x = self.fc3_2(x)
            # softmax被集成到交叉熵损失里面，所以在网络最后不需要添加softmax.
            return x
        if self.nn_number == 3:  # crnn-only-
            if self.nn_lstm_layer == 1:
                x1_1 = self.conv1_1(x1)
                x1_1 = self.Maxpool_2d(x1_1)
                x1_1 = self.conv1_2(x1_1)
                x1_1 = self.conv1_3(x1_1)
                # print(x1_1.size())
                # x1_1 = self.ca(x1_1) * x1_1
                # x1_1 = self.sa(x1_1) * x1_1

                x1_1 = self.adaptMaxpool_2d(x1_1)

                x1_1 = torch.flatten(x1_1, 1)
                # 将图像转为灰度图
                x1_2 = x1.cpu()
                x1_2 = color.rgb2gray(x1_2)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x1_2 = torch.from_numpy(x1_2).to(device)
                x1_2, _ = self.lstm_parallel_1(x1_2)
                x = torch.cat((x1_1, x1_2[:, -1, :]), 1)

                x = self.fc1_CRNN(x)
                # x = self.fc2(x)
                x = self.ReLU(x)
                x = self.dropout(x)
                x = self.fc3(x)
                # softmax被集成到交叉熵损失里面，所以在网络最后不需要添加softmax.
                return x
            elif self.nn_lstm_layer == 2:
                x1_1 = self.conv1_1(x1)
                x1_1 = self.Maxpool_2d(x1_1)
                x1_1 = self.conv1_2(x1_1)
                x1_1 = self.conv1_3(x1_1)
                # print(x1_1.size())
                # x1_1 = self.ca(x1_1) * x1_1
                # x1_1 = self.sa(x1_1) * x1_1

                x1_1 = self.adaptMaxpool_2d(x1_1)

                x1_1 = torch.flatten(x1_1, 1)
                # 将图像转为灰度图
                x1_2 = x1.cpu()
                x1_2 = color.rgb2gray(x1_2)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x1_2 = torch.from_numpy(x1_2).to(device)
                x1_2, _ = self.lstm_parallel_2(x1_2)
                x = torch.cat((x1_1, x1_2[:, -1, :]), 1)

                x = self.fc1_CRNN(x)
                # x = self.fc2(x)
                x = self.ReLU(x)
                x = self.dropout(x)
                x = self.fc3(x)
                # softmax被集成到交叉熵损失里面，所以在网络最后不需要添加softmax.
                return x
            elif self.nn_lstm_layer == 3:
                x1_1 = self.conv1_1(x1)
                x1_1 = self.Maxpool_2d(x1_1)
                x1_1 = self.conv1_2(x1_1)
                x1_1 = self.conv1_3(x1_1)
                # print(x1_1.size())
                # x1_1 = self.ca(x1_1) * x1_1
                # x1_1 = self.sa(x1_1) * x1_1

                x1_1 = self.adaptMaxpool_2d(x1_1)

                x1_1 = torch.flatten(x1_1, 1)
                # 将图像转为灰度图
                x1_2 = x1.cpu()
                x1_2 = color.rgb2gray(x1_2)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x1_2 = torch.from_numpy(x1_2).to(device)
                x1_2, _ = self.lstm_parallel_3(x1_2)
                x = torch.cat((x1_1, x1_2[:, -1, :]), 1)

                x = self.fc1_CRNN(x)
                # x = self.fc2(x)
                x = self.ReLU(x)
                x = self.dropout(x)
                x = self.fc3(x)
                # softmax被集成到交叉熵损失里面，所以在网络最后不需要添加softmax.
                return x
        if self.nn_number == 4:  # cnn-only-doa
            x1_1 = self.conv1_1(x1)
            x1_1 = self.Maxpool_2d(x1_1)
            x1_1 = self.conv1_2(x1_1)
            x1_1 = self.conv1_3(x1_1)
            # print(x1_1.size())
            # x1_1 = self.ca(x1_1) * x1_1
            # x1_1 = self.sa(x1_1) * x1_1

            x1_1 = self.adaptMaxpool_2d(x1_1)

            x1 = torch.flatten(x1_1, 1)

            x = self.fc1_CNN(x1)
            # x = self.fc2(x)
            x = self.ReLU(x)
            x = self.dropout(x)
            x = self.fc3(x)
            # softmax被集成到交叉熵损失里面，所以在网络最后不需要添加softmax.
            return x

