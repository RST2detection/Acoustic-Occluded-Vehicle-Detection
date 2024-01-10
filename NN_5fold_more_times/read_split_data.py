import os
import random
import shutil

import torch.cuda
# 在进行data_split之前先把数据按照之前的规则重新创建一个datasets的文件夹，把各种类别的图片对应的复制进去，
# 在这一轮的运算结束之后再删除这些文件夹即可。


def recombine(root, test_dir_number, class_number):
    # 创建新的真正用于训练的文件夹
    print('generate test %s datasets' % test_dir_number)
    array_files = ['train/', 'test/']
    for array_file in array_files:
        # 创建好所需的文件夹datasets_1/datasets/train(test)/1(23456)
        for m in range(1, class_number+1):
            final_path = root + 'datasets/' + array_file + str(m) + '/'
            if os.path.exists(final_path):  # 判断文件夹是否存在
                # 获取文件夹下的所有文件和文件夹
                contents = os.listdir(final_path)
                # 遍历文件夹下的所有内容，并逐一删除
                for item in contents:
                    item_path = os.path.join(final_path, item)  # 构造完整的路径
                    if os.path.isfile(item_path):  # 判断是否为文件
                        os.remove(item_path)  # 删除文件
                    elif os.path.isdir(item_path):  # 判断是否为文件夹
                        shutil.rmtree(item_path)  # 递归删除文件夹及其内容
                print(f'{final_path} already exists! and delete files in it')
            else:
                # print('created!')
                # print(f'{final_path} is created')
                os.makedirs(final_path)
        # 往创建好的文件夹里面复制各种类别的图片文件
    for j in range(1, 6):
        # 除了test_dir_number文件夹中的文件，其余都复制到train文件夹中
        if j == test_dir_number:  # 将测试文件夹下该编号的文件进行复制到对应的文件夹下：
            for test_number in range(1, class_number+1):  # 不同点是用的test文件夹下的图片文件。
                test_middlefile_path = root + 'test/' + str(j) + '/data/' + str(test_number) + '/'
                test_finalfile_path = root + 'datasets/test/' + str(test_number) + '/'
                test_imgfile_paths = os.listdir(test_middlefile_path)
                for tp in test_imgfile_paths:
                    old_path_test = test_middlefile_path + str(tp)
                    new_path_test = test_finalfile_path + str(test_dir_number) + '_' + str(tp)
                    shutil.copy(old_path_test, new_path_test)
        else:
            for class_number in range(1, class_number+1):  # 复制train中的图片到train文件中
                train_middlefile_path = root + 'train/' + str(j) + '/data/' + str(class_number) + '/'
                train_finalfile_path = root + 'datasets/train/' + str(class_number) + '/'
                train_imgfile_paths = os.listdir(train_middlefile_path)
                for trainp in train_imgfile_paths:
                    old_path_train = train_middlefile_path + str(trainp)
                    new_path_train = train_finalfile_path + str(j) + '_' + str(trainp)
                    shutil.copy(old_path_train, new_path_train)


def delete_datasets(root):
    datasets_path = root + 'datasets/'
    shutil.rmtree(datasets_path)


def read_split_data(root: str, test_dir_number):
    # 重组train和test文件夹
    # recombine(root, test_dir_number)
    global test_image_class, train_image_class
    assert os.path.exists(root), "dataset root: {} does not exist,".format(root)
    train_root = os.path.join(root+"datasets/train/")  # 这里有修改
    test_root = os.path.join(root+"datasets/test/")  # 这里有修改
    item_class = [cla for cla in os.listdir(train_root)]  # ['1', '2', '3', '4', '5', '6']
    item_class.sort()  # 对list中的对象进行排序。
    class_indices = dict((k, v) for v, k in enumerate(item_class))
    train1_images_path = []
    train2_images_path = []
    train_images_label = []
    test1_images_path = []
    test2_images_path = []
    test_images_label = []
    every_class_num = []
    # print(item_class)
    # 构建train类的总路径
    for cla in item_class:
        # print(cla)
        cla_path_train = os.path.join(train_root, cla)
        cla_path_test = os.path.join(test_root, cla)
        count_train = 0  # 计数器
        images = []
        train_images = []
        test_images = []
        for i in os.listdir(cla_path_train):
            # print(i)
            count_train += 1  # 只读单数图片，为什么只读单数图片？双数图片带个'(1)'吧
            if count_train % 2 == 0:
                images.append(os.path.join(cla_path_train, i))
                train_images.append(os.path.join(cla_path_train, i))
            train_image_class = class_indices[cla]
        every_class_num.append(len(train_images)*2)
        count_test = 0
        for i in os.listdir(cla_path_test):
            # print(i)
            count_test += 1
            if count_test % 2 == 0:
                images.append(os.path.join(cla_path_test, i))
                test_images.append(os.path.join(cla_path_test, i))
                test_image_class = class_indices[cla]
        every_class_num.append(len(test_images)*2)
        # 按比例随机采样验证样本
        # test_path = random.sample(images, k=int(len(images) * val_rate))
        test_path = test_images
        for img_path in images:
            if img_path in test_path:   # 如果该路径在采样的验证集样本中则存入验证集
                test1_images_path.append(img_path)
                # 去掉末尾的.jpeg即去掉图片的格式，而不是用这种以点进行分割的方法。
                # str_list = img_path.split(sep='.')
                str_list = img_path[:-5]
                # print(str_list)
                suffix = '(1)'
                # img_path2 = str_list[0] + suffix + str_list[1]
                img_path2 = str_list + suffix + '.jpeg'
                test2_images_path.append(img_path2)
                # print(test_image_class)
                test_images_label.append(test_image_class)
            else:  # 否则，加入训练集
                train1_images_path.append(img_path)
                # str_list = img_path.split(sep='.')
                str_list = img_path[:-5]
                suffix = '(1)'
                img_path2 = str_list + suffix + '.jpeg'
                train2_images_path.append(img_path2)
                train_images_label.append(train_image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training1.".format(len(train1_images_path)))
    print("{} images for training2.".format(len(train2_images_path)))
    print("{} images for test1.".format(len(test1_images_path)))
    print("{} images for test2.".format(len(test2_images_path)))
    return train1_images_path, train2_images_path, train_images_label, test1_images_path,\
           test2_images_path, test_images_label








