import numpy as np
from PIL import Image


# 添加椒盐噪声
class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)  # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0  # 椒
        img[mask == 1] = 255  # 盐
        img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
        return img


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255  # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

#   主函数使用：
#   my_transforms.AddGaussianNoise(mean=0, variance=1, amplitude=20),
#   transforms.ToTensor(),
