# WiFi-Ushape: U-shape Deep Networks are Efficient Backbones for Human Activity Understanding from Wi-Fi Signals

## Prerequisite
* numpy
* pandas
* scipy
* torch
* tqdm
* scikit-learn

## How to run
0. Download the dataset ARIL from[Joint Activity Recognition and Indoor Localization With WiFi Fingerprints](https://ieeexplore.ieee.org/abstract/document/8740950).
   
   Download the dataset WiAR from [here](https://github.com/ermongroup/Wifi_Activity_Recognition).
   
   Download the dataset HTHI from [here](https://drive.google.com/file/d/1R79ciMFIr_6GgwnJeP3EzJokiWu80hun/view?usp=sharing)
2. "git clone" this repository.
   
3. Datasets ARIL and HTHI do not require processing. Datasets ARIL and HTHI do not require processing，
   1. cd create_wiar_dataset
   2. python3 dataset_load.py
   3. python3 traintestsplit.py xxx  (xxx is an int type, indicating the round of random division)
      
4. Run bash run.sh
