import time
import shutil
import matplotlib.pyplot as plt
import torch
# from NN_5fold_more_times import my_transforms
# from NN_5fold_more_times import my_transforms
# import my_transforms
from model import C_lenet
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from torchvision import transforms
import torch.nn as nn
import numpy as np
from read_split_data import read_split_data, delete_datasets, recombine
from Dataset import MyDataSet
# from model_morelstmcells import C_lenet_morelstmcells
from PIL import Image
from my_transforms import AddSaltPepperNoise, AddGaussianNoise

# 混淆矩阵的求取用到了confusion_matrix函数，其定义如下：
def confusion_matrix(preds, labels, conf_matrix, device):
    preds = torch.argmax(preds, 1)
    labels_tempt = labels.cpu().numpy()
    # print(len(str(labels_tempt)))
    if len(str(labels_tempt)) == 1:
        array_temp = np.array([])
        print(array_temp)
        array_temp = np.append(array_temp, int(labels_tempt))
        array_temp = array_temp.astype(int)
        print(array_temp)
        labels_1 = torch.from_numpy(array_temp)
        stacked = torch.stack(
            (preds, labels_1.to(device)), dim=1
        )
        for pair in stacked:
            p, t_labels = pair.tolist()
            conf_matrix[p, t_labels] += 1
        return conf_matrix
    else:
        stacked = torch.stack(
            (preds, labels), dim=1
        )

        for pair in stacked:
            p, t_labels = pair.tolist()
            conf_matrix[p, t_labels] += 1
        return conf_matrix


# Dataloader
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(",")

    return np.asfarray(data, float)


def results_plot_acc(dir1_path, dir2_path):
    y_1 = data_read(dir1_path)
    x_1 = range(len(y_1))

    y_2 = data_read(dir2_path)
    x_2 = range(len(y_2))
    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel('acc')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_1, y_1, linewidth=1, linestyle="solid", label="train acc")
    plt.plot(x_2, y_2, linewidth=1, linestyle="solid", label="valid acc")

    plt.legend()
    plt.title('acc curve')
    # plt.show()


def results_plot_loss(dir1_path, dir2_path):
    y_1 = data_read(dir1_path)
    x_1 = range(len(y_1))

    y_2 = data_read(dir2_path)
    x_2 = range(len(y_2))
    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_1, y_1, linewidth=1, linestyle="solid", label="train loss")
    plt.plot(x_2, y_2, linewidth=1, linestyle="solid", label="valid loss")

    plt.legend()
    plt.title('acc curve')
    # plt.show()


# training and testing
def train_1fold(root, EPOCH, BATCH_SIZE, LR, test_dirnumber, nn_number, model_file_path, class_number, lstm_layer, data_augment_number):
    print('网络结构：', nn_number)
    global return_array_final, conf_matrix, test_date_set, train_data_set
    recombine(root, test_dirnumber, class_number)
    return_array = np.array([0.0, 0.0])
    return_array[0] = test_dirnumber
    time.sleep(15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    train1_images_path, train2_images_path, train_images_label, test1_images_path, \
        test2_images_path, test_images_label = read_split_data(root, test_dirnumber)
    # print(train1_images_path)
    # print(train_images_label)
    # 啥都没有增强
    data_transform_0 = {
        "train": transforms.Compose([transforms.Resize(61),
                                     # transforms.RandomCrop([28, 28]),
                                     # my_transforms.AddSaltPepperNoise(0.05),
                                     transforms.ColorJitter(0.5, 0.5, 0.5),
                                     # transforms.RandomErasing(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(61),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
    # 打开colorjitter
    data_transform_1 = {
        "train": transforms.Compose([transforms.Resize(61),
                                     # transforms.RandomCrop([28, 28]),
                                     transforms.ColorJitter(0.5, 0.5, 0.5),
                                     # my_transforms.AddSaltPepperNoise(0.05),
                                     # transforms.RandomErasing(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(61),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
    # 长方形的遮挡
    data_transform_2 = {
        "train": transforms.Compose([transforms.Resize(61),
                                     # transforms.RandomCrop([28, 28]),
                                     # transforms.ColorJitter(0.5, 0.5, 0.5),
                                     # my_transforms.AddSaltPepperNoise(0.05),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                     transforms.RandomErasing(p=1, scale=(0.0819, 0.0819), ratio=(12.2, 12.2), value=0)]),
        "val": transforms.Compose([transforms.Resize(61),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
    # 竖直的遮挡
    data_transform_3 = {
        "train": transforms.Compose([transforms.Resize(61),
                                     # transforms.RandomCrop([28, 28]),
                                     # transforms.ColorJitter(0.5, 0.5, 0.5),
                                     # my_transforms.AddSaltPepperNoise(0.05),
                                     transforms.ToTensor(),
                                     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                     transforms.RandomErasing(p=1, scale=(0.0819, 0.0819), ratio=(0.0819, 0.0819), value=0)]),
        "val": transforms.Compose([transforms.Resize(61),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
    # specaug_frequency
    data_transform_4 = {
        "train": transforms.Compose([transforms.Resize(61),
                                     # transforms.RandomCrop([28, 28]),
                                     # transforms.ColorJitter(0.5, 0.5, 0.5),
                                     # my_transforms.AddSaltPepperNoise(0.05),
                                     # my_transforms.specaug_freq(),
                                     # transforms.RandomErasing(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(61),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
    # specaug_time
    data_transform_5 = {
        "train": transforms.Compose([transforms.Resize(61),
                                     # transforms.RandomCrop([28, 28]),
                                     # transforms.ColorJitter(0.5, 0.5, 0.5),
                                     # my_transforms.AddGaussianNoise(mean=0, variance=1, amplitude=20),
                                     # transforms.RandomErasing(),
                                     # my_transforms.specaug_freq(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(61),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
    if data_augment_number == 0:
        train_data_set = MyDataSet(images_path1=train1_images_path,
                                   images_path2=train2_images_path,
                                   images_class=train_images_label,
                                   transform=data_transform_0["train"])
        test_date_set = MyDataSet(images_path1=test1_images_path,
                                  images_path2=test2_images_path,
                                  images_class=test_images_label,
                                  transform=data_transform_0["val"])
    elif data_augment_number == 1:
        train_data_set = MyDataSet(images_path1=train1_images_path,
                                   images_path2=train2_images_path,
                                   images_class=train_images_label,
                                   transform=data_transform_1["train"])
        test_date_set = MyDataSet(images_path1=test1_images_path,
                                  images_path2=test2_images_path,
                                  images_class=test_images_label,
                                  transform=data_transform_1["val"])
    elif data_augment_number == 2:
        train_data_set = MyDataSet(images_path1=train1_images_path,
                                   images_path2=train2_images_path,
                                   images_class=train_images_label,
                                   transform=data_transform_2["train"])
        test_date_set = MyDataSet(images_path1=test1_images_path,
                                  images_path2=test2_images_path,
                                  images_class=test_images_label,
                                  transform=data_transform_2["val"])
    elif data_augment_number == 3:
        train_data_set = MyDataSet(images_path1=train1_images_path,
                                   images_path2=train2_images_path,
                                   images_class=train_images_label,
                                   transform=data_transform_3["train"])
        test_date_set = MyDataSet(images_path1=test1_images_path,
                                  images_path2=test2_images_path,
                                  images_class=test_images_label,
                                  transform=data_transform_3["val"])
    elif data_augment_number == 4:
        train_data_set = MyDataSet(images_path1=train1_images_path,
                                   images_path2=train2_images_path,
                                   images_class=train_images_label,
                                   transform=data_transform_4["train"])
        test_date_set = MyDataSet(images_path1=test1_images_path,
                                  images_path2=test2_images_path,
                                  images_class=test_images_label,
                                  transform=data_transform_4["val"])
    elif data_augment_number == 5:
        train_data_set = MyDataSet(images_path1=train1_images_path,
                                   images_path2=train2_images_path,
                                   images_class=train_images_label,
                                   transform=data_transform_5["train"])
        train_data_set.__add__()
        test_date_set = MyDataSet(images_path1=test1_images_path,
                                  images_path2=test2_images_path,
                                  images_class=test_images_label,
                                  transform=data_transform_5["val"])
    train_num = len(train_data_set)
    val_num = len(test_date_set)
    batch_size = BATCH_SIZE
    # number of workers: CPU逻辑处理器的个数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers'.format(nw))
    print('nw: ', nw)
    train_loader = torch.utils.data.DataLoader(
        train_data_set, batch_size=batch_size, shuffle=True, num_workers=nw, generator=torch.Generator().manual_seed(0)
    )

    test_loader = torch.utils.data.DataLoader(
        test_date_set, batch_size=batch_size, shuffle=False, num_workers=nw, generator=torch.Generator().manual_seed(0)
    )

    cnn = C_lenet(nn_number, class_number, lstm_layer).to(device)
    # cnn = C_lenet_morelstmcells(nn_number, class_number, lstm_layer).to(device)

    # optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.8, weight_decay=0.01)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=51, gamma=0.2)

    loss_function = nn.CrossEntropyLoss().to(device)
    train_loss_list = []
    valid_loss_list = []
    valid_acc_list = []
    train_acc_list = []
    best_acc = 0.0
    best_epoch = 0
    best_confus = np.zeros((class_number, class_number))
    # 引入test_dir_number将其最佳精确值相加。
    for epoch in range(EPOCH):
        # .train()的作用是启动Batch Normalization 和 Dropout。在训练一个模型时，这个指令可以
        # 完全改变训练的结果。
        conf_matrix = torch.zeros(class_number, class_number)  # 首先定义一个分类树*分类树的空混淆矩阵
        cnn.train()
        train_acc = 0
        train_loss = 0
        # train_loss = 0
        # print('train_loss: ', train_loss)
        # train_loss_list.append(train_loss)
        j = 0
        for step1, data in enumerate(train_loader):
            j = j + 1
            images1, images2, labels = data
            optimizer.zero_grad()
            output = cnn(images1.to(device), images2.to(device))
            train_predict_y = torch.max(output, dim=1)[1]
            train_acc += torch.eq(train_predict_y, labels.to(device)).sum().item()
            loss = loss_function(output, labels.to(device))
            train_loss = train_loss + loss.item()
            loss.backward()
            optimizer.step()
        train_accurate = train_acc / train_num
        train_acc_list.append(train_accurate)
        train_loss = train_loss / train_num
        train_loss_list.append(train_loss)

        cnn.eval()
        valid_acc = 0.0
        valid_loss = 0.0
        # 将预测结果放到一维矩阵，将实际值放到一个一维矩阵，最后拼接成一个向量
        test_preds = []
        pth_name = model_file_path + str(nn_number) + '_' + str(test_dirnumber) + '_best_acc' + '.pth'  # 最佳精度的模型保存名称
        for step1, val_data in enumerate(test_loader):
            val_images1, val_images2, val_labels = val_data
            val_labels_matrix = val_labels.squeeze()
            val_labels_matrix = val_labels_matrix.cuda()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            outputs = cnn(val_images1.to(device), val_images2.to(device))
            conf_matrix = confusion_matrix(outputs, val_labels_matrix, conf_matrix, device)
            loss = loss_function(outputs, val_labels.to(device))
            valid_loss = valid_loss + loss.item()
            # loss.backward()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            valid_predict_y = torch.max(outputs, dim=1)[1]

            valid_acc += torch.eq(valid_predict_y, val_labels.to(device)).sum().item()
        val_accurate = valid_acc / val_num

        valid_acc_list.append(val_accurate)
        valid_loss = valid_loss / val_num
        valid_loss_list.append(valid_loss)
        scheduler.step()

        with open(model_file_path + "train_acc.txt", 'w') as train_ac:
            train_ac.write(str(train_acc_list))
        with open(model_file_path + "train_loss.txt", 'w') as train_los:
            train_los.write(str(train_loss_list))
        with open(model_file_path + "valid_acc.txt", 'w') as valid_ac:
            valid_ac.write(str(valid_acc_list))
        with open(model_file_path + "valid_loss.txt", 'w') as valid_los:
            valid_los.write(str(valid_loss_list))
        # 给最佳的模型绘制混淆矩阵：
        if val_accurate >= best_acc:
            best_acc = val_accurate
            return_array[1] = round(best_acc * 100, 3)
            best_epoch = epoch + 1
            torch.save(cnn.state_dict(), pth_name)
            conf_matrix = np.array(conf_matrix.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
            best_confus = conf_matrix
            corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
            per_kinds = conf_matrix.sum(axis=0)  # 抽取每个分类数据总的测试条数,axis=0表示按列相加。
            per_Jaccard_index = conf_matrix.sum(axis=0) + conf_matrix.sum(axis=1)
            # print(per_Jaccard_index)
            # print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), val_num))
            print(scheduler.get_last_lr())
            print('[epoch %d] val_acc: %.4f, train_acc: %.4f, best_acc(%d): %.4f' %
                  (epoch + 1, val_accurate, train_accurate, best_epoch, best_acc))
            print(conf_matrix)
            append_array = [round(rate * 100, 3) for rate in corrects / (per_Jaccard_index - corrects)]
            return_list = list(return_array)
            return_list.extend(append_array)
            return_array_final = np.array(return_list)
            np.set_printoptions(suppress=True)
            # 获取每种Emotion的识别准确率
            print("______每种类别总个数：", per_kinds)
            print("每种类别预测正确的个数：", corrects)
            print("每种类别的杰卡德的个数：", per_Jaccard_index - corrects)
            print("每种类别的识别准确率为：{0}".format([round(rate * 100, 2) for rate in corrects / per_kinds]))
            print("每种类别的杰卡德参数为：{0}".format(
                [round(rate * 100, 2) for rate in corrects / (per_Jaccard_index - corrects)]))
        else:
            best_acc = best_acc
            best_epoch = best_epoch
            best_confus = best_confus
            conf_matrix = np.array(conf_matrix.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
            corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
            per_kinds = conf_matrix.sum(axis=0)  # 抽取每个分类数据总的测试条数,axis=0表示按列相加。
            per_Jaccard_index = conf_matrix.sum(axis=0) + conf_matrix.sum(axis=1)
            # print(per_Jaccard_index)
            # print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), val_num))
            # print(conf_matrix)
            np.set_printoptions(suppress=True)
            print(scheduler.get_last_lr())
            print('[epoch %d] val_acc: %.4f, train_acc: %.4f, best_acc(%d): %.4f' %
                  (epoch + 1, val_accurate, train_accurate, best_epoch, best_acc))
            # 获取每种Emotion的识别准确率
            # print("______每种类别总个数：", per_kinds)
            # print("每种类别预测正确的个数：", corrects)
            # print("每种类别的杰卡德的个数：", per_Jaccard_index - corrects)
            # print("每种类别的识别准确率为：{0}".format([round(rate * 100, 2) for rate in corrects / per_kinds]))
            print("每种类别的杰卡德参数为：{0}".format(
                [round(rate * 100, 2) for rate in corrects / (per_Jaccard_index - corrects)]))

    delete_datasets(root)
    results_plot_acc(model_file_path + "train_acc.txt", model_file_path + "valid_acc.txt")
    results_plot_loss(model_file_path + "train_loss.txt", model_file_path + "valid_loss.txt")
    return return_array_final, np.round(best_confus)

# if __name__ == '__main__':
#     main()
