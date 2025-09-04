from sum_all_classes_imgs import all_classes_imgs_number_sum
from data_copy_test2traindataset import imgs_test2train_copy
from dataset1to6 import train_test_img2classes

if __name__ == '__main__':

    target_samples_path = 'D:/NLOS/NLOS_dataset/NLOS_xian/'

# 统计test文件夹下所有的tdoa_imgs和stft_imgs的数量。
    all_classes_imgs_number_sum(target_samples_path, 1)

    # 复制测试集文件到训练集;
    # 0: nodoaaug_original, 01: doaaug+nodoaaug, 02: nodoaaug+freq_mask, 03: nodoaaug+time_mask
    imgs_test2train_copy(target_samples_path, 1, '023', 1)

    # 最后再把测试集和训练集中的数据进行按照数据集1~6的划分；
    train_test_img2classes(target_samples_path, 1)