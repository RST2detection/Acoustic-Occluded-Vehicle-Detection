'''

将批量产生的TDOA结果图像与stft图像名称对应起来。

'''
import shutil
import os

# file_path_tdoa = '../middle_product/without_walker_data/exp_3_4/middle_product/tdoa_imgs/test/'
# file_path_stft = '../middle_product/without_walker_data/exp_3_4/middle_product/stft_imgs/test/'
# newpath = '../middle_product/without_walker_data/exp_3_4/datasets/datasets_cut_4/test/'


def train_test_img2classes(original_file_path, signal):
    if signal == 0:
        pass
    if signal == 1:

        array_files = ['train/', 'test/']
        for array_file in array_files:
            # 创建好所需的文件夹datasets_1/train(test)/1/data/1(23456)
            for m in range(1, 6):
                for k in range(1, 7):

                    save_path_for_classes = original_file_path + array_file + str(m) + '/' + 'data/' + str(k) + '/'

                    if os.path.exists(save_path_for_classes):  # 判断文件夹是否存在
                        print(f'{save_path_for_classes} already exists!')
                    else:
                        print(f'{save_path_for_classes} is created')
                        os.makedirs(save_path_for_classes)
            # 往创建好的文件夹里面复制各种类别的图片文件
            for j in range(1, 6):
                # 读取stft和tdoa图像文件中train和test的文件数量
                print(array_file,j)
                # data_path = multi_audio_path + array_file + str(j) + '/' + 'data/'
                file_path_tdoa = original_file_path + array_file + str(j) + '/tdoa_imgs'
                file_path_stft = original_file_path + array_file + str(j) + '/stft_imgs'
                newpath = original_file_path + array_file + str(j) + '/' + 'data/'
                # 取文件夹里的所有文件
                file_paths = os.listdir(file_path_tdoa)  # 取待统计文件夹下的所有文件
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
                        # print(i)

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
                            # print(i)

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
                            # print(i)
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
                            # print(i)
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
                            # print(i)
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
                        # print(i)
                        # print(quiet_sign)

                print('sum all classes imgs')
                print('front: ', front_sign)
                print('left_approach: ', left_approach_sign)
                print('left_leave: ', left_leave_sign)
                print('right_approach: ', right_approach_sign)
                print('right_leave: ', right_leave_sign)
                print('quiet: ', quiet_sign)
                print('sum: ', quiet_sign + right_leave_sign + right_approach_sign +
                    left_approach_sign + left_leave_sign + front_sign)

    print("end to the rename work")
