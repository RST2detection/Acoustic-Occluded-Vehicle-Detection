import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv
import os
import pandas as pd


# 1. 可以不用删除csv的表头就能直接出图
# 2. 增加了矩阵中最小值的计算，使得图像的对比度更加清晰。
def load_csv_files_and_run(file_path, sinal):
    if sinal == 0:
        print("仅调用csv_2_imgs方法，未执行")
    else:
        # 从train和test文件夹中取出待处理的文件。
        print("开始绘制tdoa图像")
        array_files = ['test/']
        for array_file in array_files:
            # 创建文件夹
            save_path_tdoa_imgs = file_path + 'tdoa_imgs/'
            if os.path.exists(save_path_tdoa_imgs):  # 判断文件夹是否存在
                print(f'{save_path_tdoa_imgs} already exists!')
            else:
                print(f'{save_path_tdoa_imgs} is created')
                os.makedirs(save_path_tdoa_imgs)

            csv_path = file_path + 'csv/'
            # 取文件夹里的所有文件
            file_paths = os.listdir(csv_path)  # 取csv文件夹下的所有文件

            # 对文件下的文件遍历执行操作
            for fp in file_paths:
                wav_file_name = None
                print(fp)
                wav_file_name = fp.split(".")
                # 移除数组中的最后一位元素(如close_1.5.wav会有两个“.”,分情况取文件名
                if len(wav_file_name) == 2:
                    wav_file_name = wav_file_name[0]
                if len(wav_file_name) == 3:
                    wav_file_name = wav_file_name[0] + '.' + wav_file_name[1]
                file_full_path = os.path.join(csv_path, fp)
                draw_tdoa_imgs(file_full_path, wav_file_name, save_path_tdoa_imgs)  # 对输入文件进行绘图
        print("tdoa图像绘制结束")


def draw_tdoa_imgs(file_full_path, wav_file_name, tdoa_imgs_path):
    import csv
    width = 61
    height = 61
    # img = Image.new('RGB', (150, 181), (255, 255, 255))  # 181行150列
    img = Image.new('RGB', (height, width), (255, 255, 255))  # 181行150列
    # print(img.size)
    # print(file_full_path)
    with open(file_full_path) as f:  # 读取文件
        reader = csv.reader(f)  # 创建阅读器
        rows = [row for row in reader]  # 按行读取
    row_max = 0.0
    global row_min
    m = 0
    for row in rows:
        # if m == 60:
        #     break
        # 求取每一行中的最大值
        if float(row[1]).__eq__(0) & float(row[2]).__eq__(1):
            pass
        else:
            del (row[0])
            # for i in range(0, 3721):
            for i in range(0, width * height):
                if float(row[i]) > row_max:
                    row_max = float(row[i])
                else:
                    pass
            row_min = row_max
            # for i in range(0, 3721):
            for i in range(0, width * height):
                if float(row[i]) < row_min:
                    row_min = float(row[i])
                else:
                    pass
            # y = np.array(rows[0], dtype=np.float32)  # 将列表转为数组，数据类型为浮点数
            # j对应行，i对应行
            for i in range(0, width):
                for j in range(0, height):
                    # with normalization
                    csv = int((float(row[i * height + j]) - row_min) / (row_max - row_min) * 255)
                    # without normalization
                    # csv = float(row[i * height + j]) * 255
                    img = np.array(img)
                    img[j, i] = [csv, csv, csv]
            newDir = tdoa_imgs_path + wav_file_name + '_' + str(m) + '.jpeg'  # 新文件

            cv.imwrite(newDir, img)
            m = m + 1
    # print(m)


# file_path = '../middle_product/8/4video/csv/'

# img_path = '../middle_product/8/4video/tdoa_imgs/'
# file_path = './datasets_resegment/middle_product/csv_train/'
# file_path = './datasets_resegment/middle_product/csv_test/'

# load_sound_files_and_run(file_path)
