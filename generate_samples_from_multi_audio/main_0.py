import multiprocessing
from sum_all_classes_imgs import all_classes_imgs_number_sum
from data_copy_test2traindataset import imgs_test2train_copy
from dataset1to6 import train_test_img2classes
from main_1 import main_1

if __name__ == '__main__':
    multi_audio_path = 'E:/close_and_away_detection/Exp_Deflt/datasets_test/'
    target_samples_path = 'E:/close_and_away_detection/Exp_Deflt/samples_test/'
    block_length = 1
    step_length = 0.01
    freqRange = [100, 12000]
    times_left_right = 8
    length_before_and_after_sightings = 2
    main_1_process = multiprocessing.Process(target=main_1,
                                            args=(multi_audio_path, target_samples_path, '1', block_length, step_length,
                                                  freqRange, length_before_and_after_sightings, times_left_right, 0))
    main_2_process = multiprocessing.Process(target=main_1,
                                            args=(multi_audio_path, target_samples_path, '2', block_length, step_length,
                                                  freqRange, length_before_and_after_sightings, times_left_right, 0))
    main_3_process = multiprocessing.Process(target=main_1,
                                            args=(multi_audio_path, target_samples_path, '3', block_length, step_length,
                                                  freqRange, length_before_and_after_sightings, times_left_right, 0))
    main_4_process = multiprocessing.Process(target=main_1,
                                            args=(multi_audio_path, target_samples_path, '4', block_length, step_length,
                                                  freqRange, length_before_and_after_sightings, times_left_right, 0))
    main_5_process = multiprocessing.Process(target=main_1,
                                            args=(multi_audio_path, target_samples_path, '5', block_length, step_length,
                                                  freqRange, length_before_and_after_sightings, times_left_right, 0))

    main_1_process.start()
    main_2_process.start()
    main_3_process.start()
    main_4_process.start()
    main_5_process.start()

    main_1_process.join()
    main_2_process.join()
    main_3_process.join()
    main_4_process.join()
    main_5_process.join()

    print(multi_audio_path)
    print('finished_csv_generated')

    print(
        "---------------------------------------------------------------------------------------------------------------------------------")

    # 统计test文件夹下所有的tdoa_imgs和stft_imgs的数量。
    all_classes_imgs_number_sum(target_samples_path, 1)

    # 复制测试集文件到训练集;
    # 0: nodoaaug_original, 01: doaaug+nodoaaug, 02: nodoaaug+freq_mask, 03: nodoaaug+time_mask
    imgs_test2train_copy(target_samples_path, 1, '023', 1)

    # 最后再把测试集和训练集中的数据进行按照数据集1~6的划分；
    train_test_img2classes(target_samples_path, 1)










