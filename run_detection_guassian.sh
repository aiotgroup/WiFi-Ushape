set -e

train_path='/smooth_label/ARIL/TrainDataset.mat'
test_path='smooth_label/ARIL/TestDataset.mat'
dataset_name='ARIL'

model_name='unet'
device_id=1


python3 train_eval.py --model_name ${model_name} --device_id 1 --epoches 100 --batch_size 32 --lr 0.00005 \
    --task "detection" --dataset_name ${dataset_name} --train_dataset_path ${train_path} \
    --test_dataset_path ${test_path} --detection_gaussian True