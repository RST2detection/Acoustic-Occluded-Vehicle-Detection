# 通过计算各类别的数量对数据进行复制然后进行数据的扩充。
'''

    1. 统计各类数据的数量，找出最多的那一类，然后对数据进行复制和重命名；
    2. 重命名方式仍然不要加下划线，而是采用加a或b前缀这种方式？
    3. 对于扩充倍数，仍然是采用手动调节的方式，不能自动计算然后生成对应的数据
    4. 将对应的图片文件夹生成到train文件中，并将相应的图片扩增并复制。
E:\\third_exp_data_produce\datasets_2s\exp3_4_2s_1000_10\stft_imgs_v1_no_au
E:\learn_abothird_exp_data_produce\datasets\tdoa_imgs
'''
import shutil
import os
import random

import numpy as np
from PIL import Image


# 即将其扩充为原来的多少倍，如400到2400，就写6
# times_left_right = 7
# times_quiet = 2
# #  这里看清楚是stft的图像还是tdoa的图像
# file_path_train = '../middle_product/without_walker_data' \
#             '/exp_3_4/middle_product/stft_imgs/train/'
# file_path_stft = './datasets_2s/exp3_4_2s_1000_10/data_augement_v2/stft_imgs/train/'
# newpath = 'E://learn_about//close_and_away_detection//' \
#           'occluded_vehicle_acoustic_detection-master//' \
#           'third_exp_data_produce//datasets_2s//exp3_4_2s_1000_10//train//'


def imgs_test2train_copy(original_file_path, times_copy_all, aug_number, signal):
    global image
    if signal == 0:
        pass
    if signal == 1:
        array_files = ['/tdoa_imgs/', '/stft_imgs/']
        for array_file in array_files:
            front_sign = 0
            left_approach_sign = 0
            right_approach_sign = 0
            left_leave_sign = 0
            right_leave_sign = 0
            quiet_sign = 0
            aug0_copy = 0
            aug1 = 0
            aug2 = 0
            aug3 = 0
            # 创建文件夹
            for i in range(1, 6):
                # 创建../1/tdoa_imgs的文件夹
                save_path_tdoa = original_file_path + 'train/' + str(i) + array_file

                if os.path.exists(save_path_tdoa):  # 判断文件夹是否存在
                    print(f'{save_path_tdoa} already exists!')
                else:
                    print(f'{save_path_tdoa} is created')
                    os.makedirs(save_path_tdoa)

                # 读取stft和tdoa图像文件中的文件

                test_one_data_path = original_file_path + 'test/' + str(i) + array_file
                train_one_data_path = original_file_path + 'train/' + str(i) + array_file
                # 取文件夹里的所有文件
                file_paths = os.listdir(test_one_data_path)  # 取待统计文件夹下的所有文件


                # 对文件下的文件遍历执行切片和TDOA_SRP_PHAT操作
                for fp in file_paths:
                    if fp.__contains__('front'):
                        front_sign += 1

                    if fp.__contains__('left'):
                        if fp.__contains__('appro'):
                            left_approach_sign += 1

                    if fp.__contains__('left'):  # 修改
                        if fp.__contains__('lea'):  # 修改
                            left_leave_sign += 1

                    if fp.__contains__('right'):  # 修改
                        if fp.__contains__('appro'):  # 修改
                            right_approach_sign += 1

                    if fp.__contains__('right'):  # 修改
                        if fp.__contains__('leave'):  # 修改
                            right_leave_sign += 1

                    if fp.__contains__('quie'):  # 修改
                        quiet_sign += 1

                # 判断aug_number之后开始分别进行数据增强工作。
                # 0：不进行任何的增强
                if aug_number.__contains__('0'):
                    for fp in file_paths:
                        old_path_tdoa = os.path.join(str(test_one_data_path), str(fp))
                        new_path_copy = os.path.join(str(train_one_data_path), str(fp))
                        shutil.copy(old_path_tdoa, new_path_copy)
                        aug0_copy += 1
                        # 扩充为原来的6倍
                        for j in range(times_copy_all - 1):
                            aug0_copy += 1
                            new_path_tdoa = os.path.join(
                                str(train_one_data_path), chr(j + ord('A')) + str(fp))
                            shutil.copy(old_path_tdoa, new_path_tdoa)
                # 1：我们的DOAAug
                if aug_number.__contains__('1'):
                    if array_file.__contains__('tdoa'):
                        for fp in file_paths:
                            aug1 += 1
                            old_path_tdoa = os.path.join(str(test_one_data_path), str(fp))
                            new_path_aug = os.path.join(
                                str(train_one_data_path), chr(ord('X')) + str(fp))
                            # 读取彩色图像
                            image_original = Image.open(old_path_tdoa)
                            # 将图像转换成numpy数组
                            image_array = np.array(image_original)
                            # 获取每个通道的最大值和最小值
                            min_value = np.min(image_array, axis=(0, 1))
                            max_value = np.max(image_array, axis=(0, 1))
                            # 归一化每个通道的像素值
                            normalized_image = (image_array - min_value) / (max_value - min_value)
                            # 将归一化后的数组转换回图像
                            normalized_image = (normalized_image * 255).astype(np.uint8)
                            normalized_image = Image.fromarray(normalized_image)
                            # 保存归一化后的图像
                            normalized_image.save(new_path_aug)
                    else:
                        for fp in file_paths:
                            aug1 += 1
                            old_path_tdoa = os.path.join(str(test_one_data_path), str(fp))
                            new_path_aug = os.path.join(
                                str(train_one_data_path), chr(ord('X')) + str(fp))
                            shutil.copy(old_path_tdoa, new_path_aug)
                # 2：frequency_mask
                if aug_number.__contains__('2'):
                    if array_file.__contains__('tdoa'):
                        F = 6  # 即为像素的五分之一，这个到时候得通过不断调整确定最佳的值
                        m_F = 1
                        for fp in file_paths:
                            aug2 += 1
                            old_path_tdoa = os.path.join(str(test_one_data_path), str(fp))
                            new_path_aug = os.path.join(
                                str(train_one_data_path), chr(ord('Y')) + str(fp))
                            # 读取彩色图像
                            image_original = Image.open(old_path_tdoa)
                            # 将图像转换成numpy数组
                            image_array = np.array(image_original)
                            v = image_array.shape[0]
                            # apply m_F frequency masks to the spectrogram
                            for i in range(m_F):
                                f = int(np.random.uniform(0, F))  # [0, F)
                                f0 = random.randint(0, v - f)  # [0, v - f)
                                image_array[f0:f0 + f, :, :] = 0
                                image = Image.fromarray(image_array.astype('uint8'))  # numpy转图片
                            # 保存归一化后的图像
                            image.save(new_path_aug)
                    else:
                        F = 24  # 即为像素的五分之一，这个到时候得通过不断调整确定最佳的值
                        m_F = 1
                        for fp in file_paths:
                            aug2 += 1
                            old_path_tdoa = os.path.join(str(test_one_data_path), str(fp))
                            new_path_aug = os.path.join(
                                str(train_one_data_path), chr(ord('Y')) + str(fp))
                            # 读取彩色图像
                            image_original = Image.open(old_path_tdoa)
                            # 将图像转换成numpy数组
                            image_array = np.array(image_original)
                            v = image_array.shape[0]
                            # apply m_F frequency masks to the spectrogram
                            for i in range(m_F):
                                f = int(np.random.uniform(0, F))  # [0, F)
                                f0 = random.randint(0, v - f)  # [0, v - f)
                                image_array[f0:f0 + f, :, :] = 0
                                image = Image.fromarray(image_array.astype('uint8'))  # numpy转图片
                            # 保存归一化后的图像
                            image.save(new_path_aug)
                # 3：time_mask
                if aug_number.__contains__('3'):
                    if array_file.__contains__('tdoa'):
                        T = 6  # 即为像素的五分之一，这个到时候得通过不断调整确定最佳的值
                        m_T = 1
                        for fp in file_paths:
                            aug3 += 1
                            old_path_tdoa = os.path.join(str(test_one_data_path), str(fp))
                            new_path_aug = os.path.join(
                                str(train_one_data_path), chr(ord('Z')) + str(fp))
                            # 读取彩色图像
                            image_original = Image.open(old_path_tdoa)
                            # 将图像转换成numpy数组
                            image_array = np.array(image_original)
                            t_length = image_array.shape[1]
                            # apply m_F frequency masks to the spectrogram
                            for i in range(m_T):
                                t = int(np.random.uniform(0, T))  # [0, T)
                                t0 = random.randint(0, t_length - t)  # [0, t_length - t)
                                image_array[:, :, t0:t0 + t] = 0
                                image = Image.fromarray(image_array.astype('uint8'))  # numpy转图片
                            # 保存归一化后的图像
                            image.save(new_path_aug)
                    else:
                        T = 24  # 即为像素的五分之一，这个到时候得通过不断调整确定最佳的值
                        m_T = 1
                        for fp in file_paths:
                            aug3 += 1
                            old_path_tdoa = os.path.join(str(test_one_data_path), str(fp))
                            new_path_aug = os.path.join(
                                str(train_one_data_path), chr(ord('Z')) + str(fp))
                            # 读取彩色图像
                            image_original = Image.open(old_path_tdoa)
                            # 将图像转换成numpy数组
                            image_array = np.array(image_original)
                            t_length = image_array.shape[1]
                            # apply m_F frequency masks to the spectrogram
                            for i in range(m_T):
                                t = int(np.random.uniform(0, T))  # [0, T)
                                t0 = random.randint(0, t_length - t)  # [0, t_length - t)
                                image_array[:, :, t0:t0 + t] = 0
                                image = Image.fromarray(image_array.astype('uint8'))  # numpy转图片
                            # 保存归一化后的图像
                            image.save(new_path_aug)
            print(array_file)
            print('front: ', front_sign)
            print('left_approach: ', left_approach_sign)
            print('left_leave: ', left_leave_sign)
            print('right_approach: ', right_approach_sign)
            print('right_leave: ', right_leave_sign)
            print('quiet: ', quiet_sign)
            print('aug_0_sum: ',
                  quiet_sign + right_leave_sign + right_approach_sign + left_approach_sign + left_leave_sign + front_sign)
            print('aug_1_sum: ', aug1)
            print('aug_2_sum: ', aug2)
            print('aug_3_sum: ', aug3)

            print("end to the rename work")
