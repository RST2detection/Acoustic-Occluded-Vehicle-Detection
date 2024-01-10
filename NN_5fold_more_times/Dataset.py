from PIL import Image
from torch.utils.data import Dataset

'''

    这里一个标签对应了两张图片，输入是图片1、图片2的地址以及图片1、2对应的标签，
    返回两张图片和一个标签。

'''
class MyDataSet(Dataset):
    '''自定义数据集'''

    def __init__(self, images_path1: list, images_path2: list,
                 images_class: list, transform=None):
        self.images_path1 = images_path1
        self.images_path2 = images_path2
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path1)

    def __getitem__(self, item):
        img1 = Image.open(self.images_path1[item])
        img2 = Image.open(self.images_path2[item])

        # RGB为彩色图片，L为灰度图片
        # img1 = img1.convert('L')
        # img2 = img2.convert('L')
        label = self.images_class[item]

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label
