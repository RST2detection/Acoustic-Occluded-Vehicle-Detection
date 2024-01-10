"""

将批量产生的TDOA结果图像与stft图像名称对应起来。

"""
import shutil
import os

file_path_tdoa = '../middle_product/without_walker_data/exp_3_4/middle_product/tdoa_imgs/test/'
file_path_stft = '../middle_product/without_walker_data/exp_3_4/middle_product/stft_imgs/test/'
newpath = '../middle_product/without_walker_data/exp_3_4/datasets/datasets_cut_4/test/'


def imgs_number_statistics(file_path, signal):
    array_files = ['train/', 'test/']
    for array_file in array_files:
        # 读取stft和tdoa图像文件中train和test的文件数量

        data_path = file_path + 'toda_imgs/' + array_file
        # 取文件夹里的所有文件
        file_paths = os.listdir(data_path)  # 取待统计文件夹下的所有文件
        front_sign = 0
        left_approach_sign = 0
        right_approach_sign = 0
        left_leave_sign = 0
        right_leave_sign = 0
        quiet_sign = 0
        i = 0

        # 对文件下的文件遍历执行切片和TDOA_SRP_PHAT操作
        for fp in file_paths:
            wav_file_name = None
            old_name = os.path.join(str(file_path_tdoa), fp)

            if fp.__contains__('front'):
                old_path_tdoa = os.path.join(str(file_path_tdoa), str(fp))
                old_path_stft = os.path.join(str(file_path_stft), str(fp))
                new_path_tdoa = os.path.join(
                    newpath, '1/1_' + str(front_sign) + '.jpeg')
                new_path_stft = os.path.join(
                    newpath, '1/1_' + str(front_sign) + '(1)' + '.jpeg')
                shutil.copy(old_path_tdoa, new_path_tdoa)
                shutil.copy(old_path_stft, new_path_stft)
                front_sign += 1
                i += 1
                print(i)

            if fp.__contains__('left'):
                if fp.__contains__('appro'):
                    old_path_tdoa = os.path.join(str(file_path_tdoa), str(fp))
                    old_path_stft = os.path.join(str(file_path_stft), str(fp))
                    new_path_tdoa = os.path.join(
                        newpath, '2/2_' + str(left_approach_sign) + '.jpeg')
                    new_path_stft = os.path.join(
                        newpath, '2/2_' + str(left_approach_sign) + '(1)' + '.jpeg')
                    shutil.copy(old_path_tdoa, new_path_tdoa)
                    shutil.copy(old_path_stft, new_path_stft)
                    left_approach_sign += 1
                    i += 1
                    print(i)

            if fp.__contains__('left'):  # 修改
                if fp.__contains__('lea'):  # 修改
                    old_path_tdoa = os.path.join(str(file_path_tdoa), str(fp))
                    old_path_stft = os.path.join(str(file_path_stft), str(fp))
                    # 修改
                    new_path_tdoa = os.path.join(
                        newpath, '3/3_' + str(left_leave_sign) + '.jpeg')
                    # 修改
                    new_path_stft = os.path.join(
                        newpath, '3/3_' + str(left_leave_sign) + '(1)' + '.jpeg')
                    shutil.copy(old_path_tdoa, new_path_tdoa)
                    shutil.copy(old_path_stft, new_path_stft)
                    # 修改
                    left_leave_sign += 1
                    i += 1
                    print(i)
                    # print(left_leave_sign)

            if fp.__contains__('right'):  # 修改
                if fp.__contains__('appro'):  # 修改
                    old_path_tdoa = os.path.join(str(file_path_tdoa), str(fp))
                    old_path_stft = os.path.join(str(file_path_stft), str(fp))
                    # 修改
                    new_path_tdoa = os.path.join(
                        newpath, '4/4_' + str(right_approach_sign) + '.jpeg')
                    # 修改
                    new_path_stft = os.path.join(
                        newpath, '4/4_' + str(right_approach_sign) + '(1)' + '.jpeg')
                    shutil.copy(old_path_tdoa, new_path_tdoa)
                    shutil.copy(old_path_stft, new_path_stft)
                    # 修改
                    right_approach_sign += 1
                    i += 1
                    print(i)
                    # print(right_approach_sign)

            if fp.__contains__('right'):  # 修改
                if fp.__contains__('leave'):  # 修改
                    old_path_tdoa = os.path.join(str(file_path_tdoa), str(fp))
                    old_path_stft = os.path.join(str(file_path_stft), str(fp))
                    # 修改
                    new_path_tdoa = os.path.join(
                        newpath, '5/5_' + str(right_leave_sign) + '.jpeg')
                    # 修改
                    new_path_stft = os.path.join(
                        newpath, '5/5_' + str(right_leave_sign) + '(1)' + '.jpeg')
                    shutil.copy(old_path_tdoa, new_path_tdoa)
                    shutil.copy(old_path_stft, new_path_stft)
                    # 修改
                    right_leave_sign += 1
                    i += 1
                    print(i)
                    # print(right_leave_sign)

            if fp.__contains__('quie'):  # 修改
                old_path_tdoa = os.path.join(str(file_path_tdoa), str(fp))
                old_path_stft = os.path.join(str(file_path_stft), str(fp))
                # 修改
                new_path_tdoa = os.path.join(
                    newpath, '6/6_' + str(quiet_sign) + '.jpeg')
                # 修改
                new_path_stft = os.path.join(
                    newpath, '6/6_' + str(quiet_sign) + '(1)' + '.jpeg')
                shutil.copy(old_path_tdoa, new_path_tdoa)
                shutil.copy(old_path_stft, new_path_stft)
                # 修改
                quiet_sign += 1
                i += 1
                print(i)
                # print(quiet_sign)
    print()
    print("end to the rename work")
