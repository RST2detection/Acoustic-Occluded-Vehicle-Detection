# 通过计算各类别的数量对数据进行复制然后进行数据的扩充。
'''

    1. 统计各类数据的数量，找出最多的那一类，然后对数据进行复制和重命名；
    2. 重命名方式仍然不要加下划线，而是采用加a或b前缀这种方式？
    3. 对于扩充倍数，仍然是采用手动调节的方式，不能自动计算然后生成对应的数据
E:\\third_exp_data_produce\datasets_2s\exp3_4_2s_1000_10\stft_imgs_v1_no_au
E:\learn_abothird_exp_data_produce\datasets\tdoa_imgs
'''
import shutil
import os
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


def all_classes_imgs_number_sum(original_file_path, signal):

    i = 0
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
            for j in range(1, 6):
                # 读取stft和tdoa图像文件中train和test的文件数量

                data_path = original_file_path + 'test/' + str(j) + array_file
                # 取文件夹里的所有文件
                file_paths = os.listdir(data_path)  # 取待统计文件夹下的所有文件

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

                    if fp.__contains__('front'):
                        # front_sign += 1
                        pass

            # if fp.__contains__('left'):
            #     if fp.__contains__('appro'):
            #         old_path_tdoa = os.path.join(str(file_path_train), str(fp))
            #         # 扩充为原来的6倍,0-4,复制了5次
            #         for j in range(times_left_right-1):
            #             new_path_tdoa = os.path.join(
            #                 str(file_path_train), chr(j+ord('A')) + str(fp))
            #             shutil.copy(old_path_tdoa, new_path_tdoa)
            #             left_approach_sign += 1
            #
            # if fp.__contains__('left'):  # 修改
            #     if fp.__contains__('lea'):  # 修改
            #         old_path_tdoa = os.path.join(str(file_path_train), str(fp))
            #         # 扩充为原来的6倍
            #         for j in range(times_left_right-1):
            #             new_path_tdoa = os.path.join(
            #                 str(file_path_train), chr(j + ord('A')) + str(fp))
            #             shutil.copy(old_path_tdoa, new_path_tdoa)
            #             left_leave_sign += 1
            #         # print(left_leave_sign)
            #
            # if fp.__contains__('right'):  # 修改
            #     if fp.__contains__('appro'):  # 修改
            #         old_path_tdoa = os.path.join(str(file_path_train), str(fp))
            #         # 扩充为原来的6倍
            #         for j in range(times_left_right-1):
            #             new_path_tdoa = os.path.join(
            #                 str(file_path_train), chr(j + ord('A')) + str(fp))
            #             shutil.copy(old_path_tdoa, new_path_tdoa)
            #             right_approach_sign += 1
            #         # print(right_approach_sign)
            #
            # if fp.__contains__('right'):  # 修改
            #     if fp.__contains__('leave'):  # 修改
            #         old_path_tdoa = os.path.join(str(file_path_train), str(fp))
            #         # 扩充为原来的6倍
            #         for j in range(times_left_right - 1):
            #             new_path_tdoa = os.path.join(
            #                 str(file_path_train), chr(j + ord('A')) + str(fp))
            #             shutil.copy(old_path_tdoa, new_path_tdoa)
            #             right_leave_sign += 1
            #         # print(right_leave_sign)
            #
            # if fp.__contains__('quie'):  # 修改
            #     old_path_tdoa = os.path.join(str(file_path_train), str(fp))
            #     # 扩充为原来的6倍
            #     for j in range(times_quiet - 1):
            #         new_path_tdoa = os.path.join(
            #             str(file_path_train), chr(j + ord('A')) + str(fp))
            #         shutil.copy(old_path_tdoa, new_path_tdoa)
            #         quiet_sign += 1
            print(array_file)
            print('sum all classes imgs')
            print('front: ', front_sign)
            print('left_approach: ', left_approach_sign)
            print('left_leave: ', left_leave_sign)
            print('right_approach: ', right_approach_sign)
            print('right_leave: ', right_leave_sign)
            print('quiet: ', quiet_sign)
            print('sum: ', quiet_sign+right_leave_sign+right_approach_sign+left_approach_sign+left_leave_sign+front_sign)
        print("end to the rename work")
