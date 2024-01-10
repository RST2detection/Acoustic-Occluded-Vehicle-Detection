import os
import soundfile as sf
import numpy as np


# 从左侧靠近的车辆进行分割
def save_multi_sign_right(i, points, new_signal, sr1, file_name, multi_save_path):
    # 如果从左边靠近
    if i == 0:
        new_path = os.path.join(multi_save_path + file_name + str(points) + '_' + 'right_approach.wav')
        # print(new_path)
        sf.write(new_path, new_signal, sr1)
    if i == 1:
        new_path = os.path.join(multi_save_path + file_name + str(points) + '_' + 'front.wav')
        # print(new_path)
        sf.write(new_path, new_signal, sr1)
    if i == 2:
        new_path = os.path.join(multi_save_path + file_name + str(points) + '_' + 'left_leave.wav')
        # print(new_path)
        sf.write(new_path, new_signal, sr1)


def save_single_sign_right(i, points, new_signal, sr1, file_name, single_save_path):
    # 如果从左边靠近
    if i == 0:
        new_path = os.path.join(single_save_path + file_name + str(points) + '_' + 'right_approach.wav')
        # print(new_path)
        sf.write(new_path, new_signal, sr1)
    if i == 1:
        new_path = os.path.join(single_save_path + file_name + str(points) + '_' + 'front.wav')
        # print(new_path)
        sf.write(new_path, new_signal, sr1)
    if i == 2:
        new_path = os.path.join(single_save_path + file_name + str(points) + '_' + 'left_leave.wav')
        # print(new_path)
        sf.write(new_path, new_signal, sr1)


# 三个参数：文件夹总的地址，出现前及消失后的时间节点长度，安静段的时间长度。
def multi_single_audio_cut(multi_audio_file_path, target_samples_file_path, time_length, quiet_time_point, signal):
    global points_start_left_approach_left, points_end_left_approach_left
    if signal == 0:
        print("仅调用cut_wav_2s方法，未执行")
    else:
        array_time_split = []
        print("单、多通道音频分割")
        # 从train和test文件夹中取出待处理的文件。
        array_files = ['test/']
        for array_file in array_files:
            # 创建文件夹
            save_path_single = target_samples_file_path + 'single_cut/'
            save_path_multi = target_samples_file_path + 'multi_cut/'

            if os.path.exists(save_path_single):  # 判断文件夹是否存在
                print(f'{save_path_single} already exists!')
            else:
                print(f'{save_path_single} is created')
                os.makedirs(save_path_single)
            if os.path.exists(save_path_multi):  # 判断文件夹是否存在
                print(f'{save_path_multi} already exists!')
            else:
                print(f'{save_path_multi} is created')
                os.makedirs(save_path_multi)

            wav_path = multi_audio_file_path + 'wav/'
            # 取文件夹里的所有文件
            multi_audio_files = os.listdir(wav_path)  # 取wav文件夹下的所有文件
            # 对文件下的文件遍历执行操作 0926_1354_44_54000_17.9_right_3.632_5.4_2.858.wav
            # 待修改参数t0
            for fp in multi_audio_files:
                # 定义时间序列
                # 0926_1354_44_54000_17.9_right_3.632_5.4_2.858
                wav_file_name = fp.replace('.wav', '')
                print(fp)
                file_name = fp.split('.')[0].rsplit('_', 2)[0]
                file_name = file_name + '_'
                array_time_split = [float(wav_file_name.split('_')[6]),
                                    float(wav_file_name.split('_')[7])]
                if wav_file_name.__contains__('left'):  # 左侧靠近（后面2s)，前面，右侧远离(前面2s)
                    signal_1, sr1 = sf.read(wav_path + fp)  # 调用soundfile载入音频
                    single_signal = signal_1[:, 0]
                    # print(single_signal.shape)
                    if array_time_split[0] > time_length:
                        points_start_left_approach_left = compute(sr1, array_time_split[0] - time_length)
                        points_end_left_approach_left = compute(sr1, array_time_split[0]) + 5
                    else:
                        points_start_left_approach_left = 0
                        points_end_left_approach_left = compute(sr1, array_time_split[0]) + 5
                    single_signal_1_left_approach_left = single_signal[points_start_left_approach_left: points_end_left_approach_left]
                    multi_signal_1_left_approach_left = signal_1[points_start_left_approach_left: points_end_left_approach_left]
                    save_single_sign_left(0, compute(sr1, time_length), single_signal_1_left_approach_left, sr1, file_name,
                                      save_path_single)
                    save_multi_sign_left(0, compute(sr1, time_length), multi_signal_1_left_approach_left, sr1, file_name,
                                     save_path_multi)

                    point_start_front_left = compute(sr1, array_time_split[0])
                    points_end_front_left = compute(sr1, array_time_split[1])
                    single_signal_1_front_left = single_signal[point_start_front_left: points_end_front_left]
                    multi_signal_1_front_left = signal_1[point_start_front_left: points_end_front_left]
                    save_single_sign_left(1, compute(sr1, array_time_split[1]-array_time_split[0]), single_signal_1_front_left, sr1, file_name,
                                      save_path_single)
                    save_multi_sign_left(1, compute(sr1, array_time_split[1]-array_time_split[0]), multi_signal_1_front_left, sr1, file_name,
                                     save_path_multi)

                    point_start_right_leave_left = compute(sr1, array_time_split[1])
                    points_end_right_leave_left = compute(sr1, array_time_split[1] + time_length) + 5
                    single_signal_1_right_leave_left = single_signal[
                                                   point_start_right_leave_left: points_end_right_leave_left]
                    multi_signal_1_right_leave_left = signal_1[point_start_right_leave_left: points_end_right_leave_left]
                    save_single_sign_left(2, compute(sr1, time_length), single_signal_1_right_leave_left, sr1, file_name,
                                      save_path_single)
                    save_multi_sign_left(2, compute(sr1, time_length), multi_signal_1_right_leave_left, sr1, file_name,
                                     save_path_multi)
                elif wav_file_name.__contains__('right'):
                    signal_1, sr1 = sf.read(wav_path + fp)  # 调用soundfile载入音频
                    single_signal = signal_1[:, 0]
                    if array_time_split[0] > time_length:
                        points_start_right_approach_right = compute(sr1, array_time_split[0] - time_length)
                        points_end_right_approach_right = compute(sr1, array_time_split[0]) + 5
                    else:
                        points_start_right_approach_right = 0
                        points_end_right_approach_right = compute(sr1, array_time_split[0]) + 5
                    single_signal_1_right_approach_right = single_signal[points_start_right_approach_right: points_end_right_approach_right]
                    multi_signal_1_right_approach_right = signal_1[points_start_right_approach_right: points_end_right_approach_right]
                    save_single_sign_right(0, compute(sr1, time_length), single_signal_1_right_approach_right, sr1, file_name,
                                       save_path_single)
                    save_multi_sign_right(0, compute(sr1, time_length), multi_signal_1_right_approach_right, sr1, file_name,
                                      save_path_multi)

                    point_start_front_right = compute(sr1, array_time_split[0])
                    points_end_front_right = compute(sr1, array_time_split[1])
                    single_signal_1_front_right = single_signal[point_start_front_right: points_end_front_right]
                    multi_signal_1_front_right = signal_1[point_start_front_right: points_end_front_right]
                    save_single_sign_right(1, compute(sr1, array_time_split[1]-array_time_split[0]), single_signal_1_front_right, sr1, file_name,
                                       save_path_single)
                    save_multi_sign_right(1, compute(sr1, array_time_split[1]-array_time_split[0]), multi_signal_1_front_right, sr1, file_name,
                                      save_path_multi)

                    point_start_left_leave_right = compute(sr1, array_time_split[1])
                    points_end_left_leave_right = compute(sr1, array_time_split[1] + time_length) + 5
                    single_signal_1_left_leave_right = single_signal[
                                                   point_start_left_leave_right: points_end_left_leave_right]
                    multi_signal_1_left_leave_right = signal_1[point_start_left_leave_right: points_end_left_leave_right]
                    save_single_sign_right(2, compute(sr1, time_length), single_signal_1_left_leave_right, sr1, file_name,
                                       save_path_single)
                    save_multi_sign_right(2, compute(sr1, time_length), multi_signal_1_left_leave_right, sr1, file_name,
                                      save_path_multi)
                elif wav_file_name.__contains__('quiet'):
                    signal_1, sr1 = sf.read(wav_path + fp)  # 调用soundfile载入音频
                    point_start = 0
                    i = 0
                    time_point = quiet_time_point
                    # 对三段音频数据进行多通道的分离、单通道的分离。这个也要有单通道的
                    # single_signal[i: j]取i到j行，范围是左闭右开。
                    single_signal = signal_1[:, 0]
                    points_end = compute(sr1, time_point) + point_start
                    single_signal_1_quiet = single_signal[point_start: points_end]
                    # 将train和test文件分别保存，save_path里已经有了train和test的字段。
                    new_path = os.path.join(save_path_single + file_name + str(compute(sr1, time_point)) + '_' + 'quiet.wav')
                    sf.write(new_path, single_signal_1_quiet, sr1)
                    # 取这一行以及这一行以后的n行
                    multi_signal_1_quiet = signal_1[point_start: points_end]
                    new_path = os.path.join(save_path_multi + file_name + str(compute(sr1, time_point)) + '_' + 'quiet.wav')
                    # print(new_path)
                    sf.write(new_path, multi_signal_1_quiet, sr1)
        print("单、多通道音频生成结束！")


def compute(sr, time_point):
    range = int(time_point * sr)
    return range


def save_multi_sign_left(i, points, new_signal, sr1, file_name, multi_save_path):
    # 如果从左边靠近
    if i == 0:
        new_path = os.path.join(multi_save_path + file_name + str(points) + '_' + 'left_approach.wav')
        # print(new_path)
        sf.write(new_path, new_signal, sr1)
    if i == 1:
        new_path = os.path.join(multi_save_path + file_name + str(points) + '_' + 'front.wav')
        # print(new_path)
        sf.write(new_path, new_signal, sr1)
    if i == 2:
        new_path = os.path.join(multi_save_path + file_name + str(points) + '_' + 'right_leave.wav')
        # print(new_path)
        sf.write(new_path, new_signal, sr1)


def save_single_sign_left(i, points, new_signal, sr1, file_name, single_save_path):
    # 如果从左边靠近
    if i == 0:
        new_path = os.path.join(single_save_path + file_name + str(points) + '_' + 'left_approach.wav')
        # new_path = os.path.join(new_dir_path_single+file_name+str(points)+'_'+'quiet.wav')
        # print(new_path)
        sf.write(new_path, new_signal, sr1)
    if i == 1:
        new_path = os.path.join(single_save_path + file_name + str(points) + '_' + 'front.wav')
        # print(new_path)
        sf.write(new_path, new_signal, sr1)
    if i == 2:
        new_path = os.path.join(single_save_path + file_name + str(points) + '_' + 'right_leave.wav')
        # print(new_path)
        sf.write(new_path, new_signal, sr1)

