set -e

train_path='wifi/ARIL/train_data.mat'
test_path='wifi/ARIL/test_data.mat'
dataset_name='ARIL'
task='detection'
model_name='fcn'
device_id=1


python3 train_eval.py --model_name ${model_name} --device_id 1 --epoches 100 --batch_size 32 --lr 0.00005 \
    --task ${task} --dataset_name ${dataset_name} --train_dataset_path ${train_path} \
    --test_dataset_path ${test_path} --detection_gaussian "No"