# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# from cut_wav_2s import multi_single_audio_cut
from TDOA_2_csv import csv_load_sound_files_and_run
from csv_2_imgs import load_csv_files_and_run
from STFT_images_generated import load_singlecut_files_and_run
from data_only_copy_aug import imgs_number_statistics
from sum_all_classes_imgs import all_classes_imgs_number_sum
from data_copy_test2traindataset import imgs_test2train_copy
from dataset1to6 import train_test_img2classes
from cut_wav import multi_single_audio_cut

if __name__ == '__main__':

    # 只需要有一个dataset_1/test/1/wav文件，把这个test文件夹放好就行了，其他文件夹都不需要。
    original_file_path = 'E:/close_and_away_detection/Exp_Deflt/ourmethod_on_ourdata_TUDdata/ourdataset' \
                       '/data_with_absolute_dB/datasets_test_amplitude/'
    array_i = ['1', '2', '3', '4', '5']
    # array_i = ['2']

    # block_lengh = 1

    # 先对test进行处理，在test文件夹中生成完成之后，再复制到train的文件夹，进行相应的数据扩充。
    for i in array_i:
        print(i)
        # 首先把wav文件按照一定的长度t0 进行分割。所有方法的最后一位为标志位，1为执行，0为不执行。
        file_path = original_file_path + 'test/' + str(i) + '/'  # 创建一个总的文件夹，把wav里的train和test文件放好就行。
        multi_single_audio_cut(file_path, 2, 15, 0)  # 文件地址，距离出现和离开的点的时间长度，quiet段的时间长度。

        # multi音频生成CSV文件
        block_length = 1
        step_length = 0.01
        freqRange = [100, 6000]
        times_left_right = 8
        # .wav文件夹生成csv文件。
        csv_load_sound_files_and_run(file_path, block_length, step_length, freqRange, times_left_right, 0)
        # CSV文件生成tdoa图片
        load_csv_files_and_run(file_path, 0)
        # single音频生成stft图片, ()
        load_singlecut_files_and_run(file_path, block_length, step_length, times_left_right, 1)  # 加载待处理.wav文件夹并生成全部STFT文件。
        # 统计test文件夹中各类图片的数量，用来复制时的操作，以及判断对错。
        imgs_number_statistics(file_path, 0)

    # 统计test文件夹下所有的tdoa_imgs和stft_imgs的数量。
    all_classes_imgs_number_sum(original_file_path, 0)

    # 复制测试集文件到训练集;
    # multi_audio_path, times_left_right, times_quiet, signal
    # 0: nodoaaug_original, 01: doaaug+nodoaaug, 02: nodoaaug+freq_mask, 03: nodoaaug+time_mask
    imgs_test2train_copy(original_file_path, 1, '023', 0)

    # 最后再把测试集和训练集中的数据进行按照数据集1~6的划分；
    train_test_img2classes(original_file_path, 0)










