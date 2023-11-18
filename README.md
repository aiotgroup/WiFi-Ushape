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

1. "git clone" this repository.

2. Datasets ARIL and HTHI do not require processing. Datasets ARIL and HTHI do not require processing，
   1. unzip WiAR dataset and `cd create_wiar_dataset`
   2. run `python load_data.py` to get `csi_amp_all.mat`
   3. run `python traintestsplit.py <index>`  (`index` is an int type, indicating the round of random division)
   4. get `TestDataset1.mat` and `TrainDataset1.mat`

3. Run bash run.sh (If you want to run Gaussian mode detection, please 'bash run_detection_gaussian.sh')

### Input Parameters:

```python
python train_eval.py --model_name <model_name> --task <task> --dataset_name <dataset_name>
```

- `--model_name`: choose between `unet`, `unetpp` and `fcn`
- `--task`: choose between `classify`, `detection`, and `segment`
- `--dataset_name`: choose between `HTHI`, `WiAR` and `ARIL`

Please note that when the `dataset_name` is set to `HTHI`, the `task` parameter can only be set to `detection`.

## gaussian smooth label

`run gaussian_smooth_label.py `


## Citation
If this helps your research, please cite our [paper](https://ieeexplore.ieee.org/document/10286020).

    @article{wang2023wifiushape,
     title={U-Shape Networks are Unified Backbones for Human Action Understanding from Wi-Fi Signals},
     author={Wang, Fei and Gao, Yiao and Lan, Bo and Ding, Han and Shi, Jingang and Han, Jinsong},
     journal={IEEE Internet of Things Journal},
     year={2023},
     publisher={IEEE}
     }

