import random

import numpy as np
from PIL import Image


# 添加椒盐噪声
class specaug_freq(object):

    def __init__(self, density=0):
        self.density = density
        self.F, self.m_F, self.T, self.p, self.m_T = 12, 1, 61, 1.0, 1

    def __call__(self, img):
        img = np.array(img)  # 图片转numpy
        v = img.shape[1]  # no. of mel bins, 61

        # apply m_F frequency masks to the mel spectrogram
        for i in range(self.m_F):
            f = int(np.random.uniform(0, self.F))  # [0, F)
            f0 = random.randint(0, v - f)  # [0, v - f)
            img[:, f0:f0 + f, :, :] = 0
            img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
        return img

        # img = np.array(img)  # 图片转numpy
        # h, w, c = img.shape
        # Nd = self.density
        # Sd = 1 - Nd
        # mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
        # mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
        # img[mask == 0] = 0  # 椒
        # img[mask == 1] = 255  # 盐
        # img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
        # return img


'''
    W   : Time Warp parameter
    F   : Frequency Mask parameter
    m_F : Number of Frequency masks
    T   : Time Mask parameter
    p   : Parameter for calculating upper bound for time mask
    m_T : Number of time masks
'''


class specaug_time(object):
    def __init__(self, density=0):
        self.density = density
        self.F, self.m_F, self.T, self.p, self.m_T = 12, 1, 12, 1.0, 1

    def __call__(self, img):
        img = np.array(img)  # 图片转numpy
        tau = img.shape[2]  # time frames
        # apply m_T time masks to the mel spectrogram
        for i in range(self.m_T):
            t = int(np.random.uniform(0, self.T))  # [0, T)
            t0 = random.randint(0, tau - t)  # [0, tau - t)
            img[:, :, t0:t0 + t, :] = 0
            img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
        return img
