# Acoustic-Occluded-Vehicle-Detection
Code and newly created dateset of paper "Acoustic Non-line-of-sight Vehicle Approaching and Leaving Detection”.

[Paper](https://ieeexplore.ieee.org/document/10415313)

## Datasets
Download the dataset and add the unzipped dataset folder address to 'multi_audio_path' in generated_samples_from_multi_audio/main_0.py and customize the save address for the generated audio samples in 'target_samples_path'. Then run main_0.py to generate the samples for training and testing, it takes a while to generate the samples, please be patient.

Our full datasets can be downloaded from [here](https://pan.baidu.com/s/1z1hGTyfptad_Qwa_4F4tiA) Code: nlos

## Installation
Install the required version of environment according to requirements.txt, but there are a lot of environments inside the requirements, so it is recommended to install only the ones that are required in the code.

## Train
1. First, generate samples for training and testing:
In the ‘Acoustic-Occluded-Vehicle-Detection/generate_samples_from_multi_audio’ folder, modify the `target_samples_path` parameter in both `main_0.py` and `main_2.py` to point to the downloaded NLOS dataset location, such as `D:/NLOS/NLOS_dataset/NLOS_xian/`. When generating samples, run the ‘main_0.py’ file first, followed by the ‘main_2.py’ file. Ensure ‘main_0.py’ completes execution before running ‘main_2.py’, otherwise errors may occur. Sample generation requires significant time, typically ranging from 18 to 48 hours depending on computer performance.

2. Model Training and Testing:
In the ‘main_train_start.py’ file located within the ‘NN_5fold_more_times’ folder, modify the address parameter within the train_main() method to point to the folder containing the generated samples. Adjustments to other parameters are detailed in the code comments. Run ‘main_train_start.py’ to perform model training and testing.

3. If you have any questions about operation, feel free to ask.

