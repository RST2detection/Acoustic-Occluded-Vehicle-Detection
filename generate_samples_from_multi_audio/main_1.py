from cut_wav import multi_single_audio_cut
from TDOA_2_csv import csv_load_sound_files_and_run
from csv_2_imgs import load_csv_files_and_run
from STFT_images_generated import load_singlecut_files_and_run
from data_only_copy_aug import imgs_number_statistics

def main_1(multi_audio_path, target_samples_path, array_i, block_length, step_length, freqRange,
           length_before_and_leave_sightings, times_left_right, signal):
    if signal == 0:
        pass
    else:
        for i in array_i:
            print(i)
            # 首先把wav文件按照一定的长度t0 进行分割。所有方法的最后一位为标志位，1为执行，0为不执行。
            multi_audio_file_path = multi_audio_path + 'test/' + str(i) + '/'  # 创建一个总的文件夹，把wav里的train和test文件放好就行。
            target_samples_file_path = target_samples_path + 'test/' + str(i) + '/'  # 创建一个总的文件夹，把wav里的train和test文件放好就行。
            multi_single_audio_cut(multi_audio_file_path, target_samples_file_path, length_before_and_leave_sightings,
                                   15, 1)  # 文件地址，距离出现和离开的点的时间长度，quiet段的时间长度。
            # 加载待处理.wav文件夹并生成全部csv文件。
            csv_load_sound_files_and_run(target_samples_file_path, block_length, step_length,
                                         freqRange, times_left_right, 1)
            # CSV文件生成tdoa图片
            load_csv_files_and_run(target_samples_file_path, 1)
            # single音频生成stft图片,()
            load_singlecut_files_and_run(target_samples_file_path, block_length, step_length, times_left_right, 1)
            # 加载待处理.wav文件夹并生成全部STFT文件。
            # 统计test文件夹中各类图片的数量，用来复制时的操作，以及判断对错。
            imgs_number_statistics(target_samples_file_path, 1)
