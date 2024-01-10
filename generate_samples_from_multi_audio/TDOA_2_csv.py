# 重要参考网站：https://pyroomacoustics.readthedocs.io/en/pypi-release/
'''

### 实现功能：对多通道数据进行TDOA-SRP-PHAT估计，保存生成的csv数据及可视化图像。

1.

'''
import numpy as np
import pyroomacoustics as pra
from scipy import signal
import os
import scipy.io.wavfile as wavf
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd
import math


def extractSRPFeature(audio_data, sampleRate, mic_Array, freqRange):
    resolution = 61
    # nfft = 1024
    nfft = 256
    L = 61
    # freqRange = [2000, 6000]
    # freqRange = [100, 6000]
    # freqRange = [100, 6000]

    # generate fft lengths and filter mics and create doa algorithm 画角度网格，传参数
    doaProcessor = pra.doa.algorithms['SRP'](mic_Array.transpose(), sampleRate, nfft,
                                             azimuth=np.linspace(-90, 90, resolution, endpoint=True)
                                                     * np.pi / 180,
                                             max_four=4)  # np.linspace(start, stop, num=50)d, 在start和stop之间返回均匀间隔的数据。

    # extract the stft from parameters，stft处理
    container = []
    for i in range(audio_data.shape[1]):  # i:audio_data.shape(快拍数即时间块0.4s中的总点数，麦克风数目)
        _, _, stft = signal.stft(audio_data[:, i], sampleRate, nperseg=nfft)
        container.append(stft)
    # np.stack: 矩阵的拼接操作
    container = np.stack(container)

    # split the stft into L segments 分段进行TDOA定位
    segments = []
    delta_t = container.shape[-1] // L
    for i in range(L):
        segments.append(container[:, :, i * delta_t:(i + 1) * delta_t])
        #  pdb.set_trace()  container = [container[:, :, 0:94], container[:, :, 94:94+94]]

    # apply the doa algorithm for each specified segment according to parameters 对每一段进行TDOA定位
    feature = []
    for i in range(L):
        a = doaProcessor.locate_sources(segments[i], freq_range=freqRange)  # 第i段中的stft变换结果，送入TDOA定位。
        # grid网格，即为resolution点
        feature.append(doaProcessor.grid.values)

    return np.concatenate(feature)


# 加载麦克风阵列位置
def loadMicarray():
    ar_x = []
    ar_y = []
    # iterrate through the xml to get all locations
    root = ET.parse('../config/array_32.xml').getroot()
    for type_tag in root.findall('pos'):
        ar_x.append(type_tag.get('x'))
        ar_y.append(type_tag.get('y'))
    # set up the array vector
    micArray = np.zeros([len(ar_x), 3])
    micArray[:, 1] = ar_x
    micArray[:, 2] = ar_y
    micArrayConfig = """
  _______________________________________________________________
   Loading microphone Array with {} microphones.  
  _______________________________________________________________\n\n
        """.format(micArray.shape[0])
    print(micArrayConfig)
    return micArray


def cut_wavfile(file_full_path, wav_file_name, block_length, step_length, csv_save_path, mic_Array, freqRange, times_left_right):
    sample_rate, mic_data = wavf.read(file_full_path)  # 读取音频文件并获取采样率；
    # 循环调用SRP方法，得到多组特征。
    start_time = 0
    end_time = mic_data.shape[0] / sample_rate
    # 设计时间块长度和步长
    # block_length = 1
    # step_length = 0.1
    if wav_file_name.__contains__('front'):
        pass
    elif wav_file_name.__contains__('quiet'):
        pass
    else:
        step_length = step_length / times_left_right
    i = (end_time - block_length) / step_length
    i = math.floor(i) + 1
    print("\n%s时长%s秒，共分成 %d份，步长%s，每段时长%s秒，共%s个点\n" % (
        file_full_path, end_time, i, step_length, block_length, block_length * sample_rate))
    j = 0
    TDOA_results = []
    for s_time in np.arange(start_time, end_time - block_length + 0.00001, step_length):
        # for s_time in tqdm(np.arange(start_time, end_time - block_length + 0.000001, step_length)):
        s_time = round(s_time, 2)
        audio_data = mic_data[int(s_time * sample_rate): int((s_time + block_length) * sample_rate), :]
        feature = extractSRPFeature(audio_data, sample_rate, mic_Array, freqRange)
        j = j + 1
        TDOA_results.append(feature)
        np.concatenate(TDOA_results)

    print("%s文件应生成%s条csv数据，共生成%s条csv数据 \n" % (wav_file_name, i, j))

    pd.DataFrame(TDOA_results).to_csv(csv_save_path + wav_file_name + '.csv')
    return file_full_path


def csv_load_sound_files_and_run(target_samples_file_path, block_length, step_length, freqRange, times_left_right, signal):
    if signal == 0:
        print("仅调用tdoa_2_csv方法，未执行")
    else:
        print("生成csv文件！")
        mic_Array = loadMicarray()  # 加载麦克风坐标
        array_files = ['test/']
        for array_file in array_files:
            # 创建文件夹
            save_path_csv = target_samples_file_path + 'csv/'

            if os.path.exists(save_path_csv):  # 判断文件夹是否存在
                print(f'{save_path_csv} already exists!')
            else:
                print(f'{save_path_csv} is created')
                os.makedirs(save_path_csv)

            multi_wav_path = target_samples_file_path + 'multi_cut/'
            # 取文件夹里的所有文件并执行切片和STFT操作：
            file_paths = os.listdir(multi_wav_path)  # 取文件夹下的所有文件

            # 对文件下的文件遍历执行切片和TDOA_SRP_PHAT操作
            for fp in file_paths:
                wav_file_name = None
                wav_file_name = fp.split(".")
                # 移除数组中的最后一位元素(如close_1.5.wav会有两个“.”,分情况取文件名
                if len(wav_file_name) == 2:
                    wav_file_name = wav_file_name[0]
                else:
                    wav_file_name = wav_file_name[0]
                file_full_path = os.path.join(multi_wav_path, fp)
                cut_wavfile(file_full_path, wav_file_name, block_length, step_length, save_path_csv, mic_Array,
                            freqRange, times_left_right)  # 对输入文件进行切分

        print("csv生成结束")
    return None

# block_length = 1
# step_length = 1
# freqRange = [100, 6000]
# file_path = "../datasets_1/"  # 待处理.wav文件存放的文件夹
# csv_save_path = "../middle_product/8/4video/csv/"  # TDOA-SRP-PHAT计算结果保存的文件夹地址
# csv_load_sound_files_and_run(file_path, block_length, step_length, freqRange)  # 加载待处理.wav文件夹并生成全部STFT文件。

# 修改了srp.py中的  # 警告部分
