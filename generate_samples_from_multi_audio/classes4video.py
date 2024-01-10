import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import cv2 as cv
import os
import pandas as pd
import time
# 1. 可以不用删除csv的表头就能直接出图
# 2. 增加了矩阵中最小值的计算，使得图像的对比度更加清晰。
def load_sound_files_and_run(file_path):
    # 取文件夹里的所有文件
    file_paths = os.listdir(file_path)  # 取文件夹下的所有文件
    # 对文件下的文件遍历执行操作
    for fp in file_paths:
        wav_file_name = None
        wav_file_name = fp.split(".")
        # 移除数组中的最后一位元素(如close_1.5.wav会有两个“.”,分情况取文件名
        if len(wav_file_name) == 2:
            wav_file_name = wav_file_name[0]
        if len(wav_file_name) == 3:
            wav_file_name = wav_file_name[0] + '.' + wav_file_name[1]
        file_full_path = os.path.join(file_path, fp)
        draw_tdoa_imgs(file_full_path, wav_file_name)  # 对输入文件进行绘图


def draw_tdoa_imgs(file_full_path, wav_file_name):
    import csv
    width = 61
    height = 61
    # img = Image.new('RGB', (150, 181), (255, 255, 255))  # 181行150列
    img = Image.new('RGB', (height, width), (255, 255, 255))  # 181行150列
    print(img.size)
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
                    # print(j * 181 + i)
                    csv = int(255*float(row[i * height + j]))
                    img = np.array(img)
                    img[j, i] = [csv, csv, csv]
            newDir = img_path + wav_file_name + '_' + str(m) + '.jpeg'  # 新文件

            cv.imwrite(newDir, img)
            m = m + 1
            # print(m)


# 合成图像：
def merge_imgs(img_path):
    file_paths = os.listdir(img_path)  # 取文件夹下的所有文件
    imgs_number = len(file_paths)
    to_merge = Image.new('RGB', (61*imgs_number, 61))
    img_path_front = ((os.listdir(img_path)[0]).split('.')[0]).strip('0')
    # img = Image.open(img0_path)
    # 对文件下的文件遍历执行操作
    for j in range(0, imgs_number):
        print(j)
        img_file_name = None
        img_file_name = img_path + img_path_front + str(j) + '.jpeg'
        img_m = Image.open(img_file_name)
        to_merge.paste(img_m, (j*61, 0))

    img_full_save_path = img_path + 'tdoa_full.jpeg'
    to_merge.save(img_full_save_path)
    # 绘制线条
    img_draw = cv.imread(img_full_save_path)
    cv.line(img_draw, (point1, 0), (point1, 61), (219, 142, 27), 1)
    cv.line(img_draw, (point2, 0), (point2, 61), (219, 142, 27), 1)
    img_readline_full_save_path = img_path + 'line_tdoa_full.jpeg'
    cv.imwrite(img_readline_full_save_path, img_draw)

# 修改csv文件地址（要窗为1s,跨步为1s的定位结果）修改两个节点的时刻。
file_path = '../middle_product/1/csv_full/'

img_path = '../middle_product/1/tdoa_img_full/'
# file_path = './datasets_resegment/middle_product/csv_train/'
# file_path = './datasets_resegment/middle_product/csv_test/'
# 1. 3.583, 4.828 // 2, 4.302, 3.957 // 3. 3.225, 3.967 // 4. 3.938, 4.124
# 5. 4.126, 5.039 // 6. 3.579, 4.515 // 7. 3.763, 4.336 // 8. 4.285, 4.703
point1 = int(3.585 * 61)
point2 = int(4.828 * 61) + point1
load_sound_files_and_run(file_path)
merge_imgs(img_path)