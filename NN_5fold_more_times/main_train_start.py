
'''
    会先在原文件下建立一个datasets文件夹，然后将所有的图片都放在这个文件夹下，

'''
import os

import numpy as np
import time

import pandas as pd

from Parallel_NN import train_1fold
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_main(datasets_path, epoch, batch_size, lr, lstm_layer, neural_structure_number, class_number,
               data_augment_number):
    # (test_fold_number, best_acc, jaccrad_index1, 2, 3, 4, 5, 6)
    global fivefold_acc_array, add_acc_array, best_confusion_sum

    # 建立用于保存准确率的Jaccard_index
    time_str = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
    model_file_path = '../results/' + time_str + '/'
    os.makedirs(model_file_path)
    txt_file_path = '../results/' + time_str + '/' + time_str + '.txt'
    np.set_printoptions(suppress=True)
    with open(txt_file_path, 'a') as f:
        f.write('\n')
        f.write(datasets_path)
        f.write('\n')
    # 原本是(1, 5)
    for j in neural_structure_number:
        if class_number == 4:
            fivefold_acc_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            add_acc_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif class_number == 6:
            fivefold_acc_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            add_acc_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        with open(txt_file_path, 'a') as f:
            f.write('neural_structure_number: %s;' % j)
            f.write('\n')
            f.write('epoch:%s;  ' % epoch)
            f.write('batchsize:%s; ' % batch_size)
            f.write('learning rate:%s; ' % lr)
            f.write('lstm_layer: %s; ' % lstm_layer)
            f.write('\n\n')
        # 每一种网络结构下的五折交叉验证。
        global best_confusion
        best_confusion_sum = np.zeros((class_number, class_number))
        for i in range(1, 6):
            best_confusion = np.zeros((class_number, class_number))
            print(
                '------------------------------------------------------------------------------------',
                j)
            test_dir_number = i
            # train_1fold(orginal_file_path, epoch, batch_size, lr, test_dir_number)
            # 引入最佳精确度并将其相加根据test_dir_number进行相加
            nn_number = j
            # def train_main(datasets_path, epoch, batch_size, lr, lstm_layer, neural_structure_number, class_number):
            best_acc_jaccard, best_confusion = train_1fold(datasets_path, epoch, batch_size, lr, test_dir_number,
                                                           nn_number, model_file_path, class_number, lstm_layer,
                                                           data_augment_number)
            best_confusion_sum = best_confusion_sum + best_confusion
            # 暂时没有对五折交叉验证中的最佳参数进行处理。
            print('best acc jaccard:', best_acc_jaccard)
            add_acc_array = add_acc_array + best_acc_jaccard
            fivefold_acc_array = np.vstack((fivefold_acc_array, best_acc_jaccard))

            with open(txt_file_path, 'a') as f:
                np.set_printoptions(suppress=True)
                f.write(str(i))
                f.write(str(best_acc_jaccard))
                f.write('data_augment_number: ')
                f.write(str(data_augment_number))
                f.write('\n')
                np.savetxt(f, best_confusion, header='nn_structure:%s' % nn_number, footer='', delimiter=', ')
                f.write('\n')
        with open(txt_file_path, 'a') as f:
            np.set_printoptions(suppress=True)
            f.write('\n')
            np.savetxt(f, best_confusion_sum, header='confusion matrix sum:%s' % j, footer='', delimiter=', ')
            f.write('\n')
        add_acc_array = np.round(add_acc_array / 5, 2)
        with open(txt_file_path, 'a') as f:
            f.write(str(add_acc_array))
            f.write('\n')
        print('fivefold_acc_array:', fivefold_acc_array)
        print(' ', add_acc_array)
        xls_file = model_file_path + str(j) + '_output.xls'
        df = pd.DataFrame(best_confusion_sum)
        df.to_excel(xls_file, index=False)


if __name__ == '__main__':

    train_main('E:/close_and_away_detection/Exp_Deflt/samples_test/',
               # epoch, batchsize, lr, lstm layers, network number, class_numbers, dataaug_operation
               50, 32, 0.0001, 2, [3], 6, 0)  # CRNN only, pCNN
    train_main('E:/close_and_away_detection/Exp_Deflt/samples_test/',
               # epoch, batchsize, lr, lstm layers, network number, class_numbers, dataaug_operation
               50, 32, 0.0001, 2, [1], 6, 0)  # CRNN only, pCNN

    
