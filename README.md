# Acoustic-Occluded-Vehicle-Detection
Code and newly created dateset of paper "Acoustic Non-line-of-sight Vehicle Approaching and Leaving Detection”.

[Paper](https://ieeexplore.ieee.org/document/10415313)

## Datasets
Download the dataset and add the unzipped dataset folder address to 'multi_audio_path' in generated_samples_from_multi_audio/main_0.py and customize the save address for the generated audio samples in 'target_samples_path'. Then run main_0.py to generate the samples for training and testing, it takes a while to generate the samples, please be patient.

Our full datasets can be downloaded from [here](https://pan.baidu.com/s/1z1hGTyfptad_Qwa_4F4tiA) Code: nlos

## Installation
Install the required version of environment according to requirements.txt, but there are a lot of environments inside the requirements, so it is recommended to install only the ones that are required in the code.

## Train
The sample address in the customized parameter 'target_samples_path' in the main_0.py file is used as the parameter in the train_main method in NN_5fold_more_times/main_train_start.py as 'datasets_path', and other parameters can be set as the same as in the example of the train_main method in main_train_start.py. Run main_train_start.py to train and test, it will show the evaluation metrics for each round of training and validation.


