# U-Shape Networks are Unified Backbones for Human Action Understanding from Wi-Fi Signals

## Prerequisite
* numpy
* pandas
* scipy
* torch
* tqdm
* scikit-learn

## How to run
0. Download the dataset ARIL from [its project](https://github.com/geekfeiw/apl).
   
   Download the dataset WiAR from [its project](https://github.com/ermongroup/Wifi_Activity_Recognition).
   
   Download the dataset HTHI from [here](https://drive.google.com/file/d/1R79ciMFIr_6GgwnJeP3EzJokiWu80hun/view?usp=sharing)
2. "git clone" this repository.
   
3. Datasets ARIL and HTHI do not require processing. Datasets ARIL and HTHI do not require processing，
   1. cd create_wiar_dataset
   2. python3 dataset_load.py
   3. python3 traintestsplit.py xxx  (xxx is an int type, indicating the round of random division)
      
4. Run bash run.sh


## Citation
If this helps your research, please cite our [paper](https://ieeexplore.ieee.org/document/10286020).

    @article{wang2023wifiushape,
     title={U-Shape Networks are Unified Backbones for Human Action Understanding from Wi-Fi Signals},
     author={Wang, Fei and Gao, Yiao and Lan, Bo and Ding, Han and Shi, Jingang and Han, Jinsong},
     journal={IEEE Internet of Things Journal},
     year={2023},
     publisher={IEEE}
     }
   
