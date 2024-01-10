# 获取当前时间，创建文件夹，添加每一次实验的数据。

import time

time_str = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))

txt_file_path = '../results/' + time_str + '.txt'

with open(txt_file_path, 'a') as f:
    f.write('nn_number-text_dir_number-jaccard_index')

print(time_str)




