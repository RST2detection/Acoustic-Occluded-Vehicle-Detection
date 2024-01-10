# 生成时域图，频谱图，时频图


'''

数据集制作：

1. 输入音频文件格式：close_1.wav,根据文件名称保存STFT处理图像。

2. 输出的文件地址与输入文件的close、away或quiet相关，根据文件名保存到不同的地址。

3.1 遍历存放有待处理文件的文件夹，遍历每一个文件，并运行文件切割与快速傅里叶变换处理过程；
3.2 文件按照时间切割方法；
3.3 切割后的文件快速傅里叶变换方法与图片输出

# 可修改参数
# end_time: 音频的长度，在具体的文件中计算，这里可以不设置;
# block_length: 音频块时长, 单位：秒
# step_length: 音频块切分步长，单位：秒

# stft中可修改参数
# n_fft = 256；fft的点数
# hop_length = 64；步长点数
'''

import os
import librosa
import soundfile
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import math


# 对每一段Block进行STFT处理并生成图片。
# 此处修改为240*240
def STFT(data, wav_file_name, j, stft_imgs_path):
    plt.figure(figsize=(2.4, 2.4), dpi=100)

    # 取消坐标轴
    ax = plt.subplot()
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.axis('off')  # 去掉坐标轴
    stft = librosa.stft(data, n_fft=128, hop_length=32, window='hann')  # 对block进行处理时的点数和步长确定。
    stft1 = np.abs(stft)
    # stft1 = librosa.amplitude_to_db(stft1, ref=2 * 10e-5, top_db=200)
    stft1 = librosa.amplitude_to_db(stft1, ref=2e-5, top_db=200)

    # 频谱图，stft1: 要显示的矩阵，sr:采样率，hop_length: 帧移，所有频率类型均以Hz为单位显示。
    # 频率类型：y_axis = 'linear'，'log'，'mel'，频率分别以线性、对数、mel刻度显示。'viridis_r'
    librosa.display.specshow(stft1, vmin=0, vmax=100, sr=None, hop_length=32, cmap='viridis_r')
    # plt.colorbar(format='%+2.0f dB', label='Amplitude (dB)')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.show()
    plt.savefig(stft_imgs_path + wav_file_name + "_{}.jpeg".format(j))
    plt.cla()
    plt.close("all")

    return None


def cut_wavfile(file_full_path, wav_file_name, stft_imgs_path, block_length, step_length, times_left_right):
    # 定义输入，显示输出
    mic_data, sample_rate = librosa.load(file_full_path,
                                         sr=None)
    # librosa.load()函数用于读取文件，该函数会改变声音采样频率，如果sr缺省，librosa.load()会默认以22050的采样率读取音频文件，
    # 高于该采样率的音频文件会被下采样，低于该采样率的文件会被上采样。如果希望以原始采样率读取音频文件，sr应当设为None，具体做法为y,
    # sr = librosa(filename, sr = None)
    # 循环调用STFT方法，得到图像文件。
    start_time = 0
    end_time = mic_data.size / sample_rate  # 用点数除以采样率得到音频结束的时间
    # 设置时间块长度和步长  # print("cut_file: %s的点数为: %s" % (wav_file_name, mic_data.shape))
    if wav_file_name.__contains__('front'):
        pass
    elif wav_file_name.__contains__('quiet'):
        pass
    else:
        step_length = step_length / times_left_right
    i = (end_time - block_length) / step_length
    i = math.floor(i) + 1
    print("%s时长%s秒，共分成 %d份，步长%s，每段时长%s秒，共%s个点" % (
    file_full_path, end_time, i, step_length, block_length, block_length * sample_rate))
    j = 0
    for s_time in np.arange(start_time, end_time - block_length + 0.0001, step_length):
        # for s_time in tqdm(np.arange(start_time, end_time-block_length+0.0001, step_length)):
        # if j == 296:
        #     break
        s_time = round(s_time, 2)
        audio_data = mic_data[int(s_time * sample_rate): int((s_time + block_length) * sample_rate)]
        STFT(audio_data, wav_file_name, j, stft_imgs_path)
        # if (end_time - s_time) >= block_length:  # if audio_data.shape[0] == int(block_length * sample_rate):
        #     STFT(audio_data, wav_file_name, j)
        # else:
        #     continue
        j = j + 1
    print("%s文件应生成%s张图片，共生成%s张图片 \n" % (wav_file_name, i, j))
    return file_full_path


def load_singlecut_files_and_run(file_path, block_length, step_length, times_left_right, signal):
    if signal == 0:
        print("仅调用stft方法，未执行")
    else:
        # 从train和test文件夹中取出待处理的文件。
        print("开始绘制stft图像")
        array_files = ['test/']
        for array_file in array_files:
            # 创建文件夹
            save_path_stft_imgs = file_path + 'stft_imgs/'
            if os.path.exists(save_path_stft_imgs):  # 判断文件夹是否存在
                print(f'{save_path_stft_imgs} already exists!')
            else:
                print(f'{save_path_stft_imgs} is created')
                os.makedirs(save_path_stft_imgs)

            single_audio_path = file_path + 'single_cut/'

            # 取文件夹里的所有文件并执行切片和STFT操作；
            file_paths = os.listdir(single_audio_path)  # 取single_cut文件夹下的所有文件

            raw_sounds = []  # sound_names = []

            # 对文件夹下的文件遍历执行切片和STFT操作；
            for fp in file_paths:
                wav_file_name = None
                wav_file_name = fp.split(".")
                # 移除数组中的最后一位元素(如close_1.5.wav会有两个“.”,分情况取文件名
                if len(wav_file_name) == 2:
                    wav_file_name = wav_file_name[0]
                if len(wav_file_name) == 3:
                    # wav_file_name_temp = os.path.join(wav_file_name[0], '.')
                    wav_file_name = wav_file_name[0] + '.' + wav_file_name[1]
                # print(fp)
                file_full_name = os.path.join(single_audio_path, fp)
                cut_wavfile(file_full_name, wav_file_name, save_path_stft_imgs, block_length, step_length, times_left_right)  # 对输入文件进行切分
                # raw_sounds.append(data) # sound_names.append(wav_file_name)

        print("end to the STFT work")
    return None

# block_length = 1
# step_length = 0.1
# 1000ms_10ms的test数据已生成，train数据还未生成
# file_path = '../middle_product/8/singlecut/'  # 待处理.wav文件存放的文件夹
# save_path = '../middle_product/8/4video/stft_imgs/'  # STFT图像处理结果保存的文件地址
# load_sound_files_and_run(file_path)  # 加载待处理.wav文件夹并生成全部STFT文件。
