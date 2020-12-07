# NFL Challenge


## Models

## Dataset
The data were provided by th challenge orginisers. The script to download the dataset is in ```scripts/download_data```. 

## Images preprocessing and augmentations

## Prepare environment 
1. Install anaconda
2. Run ```scripts/create_env.sh``` bash file to set up the conda environment

# Training

## Make folds
python -m src.folds.make_folds

## Create empty masks for images with no objects
python -m src.utils.make_empty_masks

## Test the train runners
python -m src.pretrain_runner --model-name "unet_se_resnext101_32x4d" --encoder "se_resnext101_32x4d" --debug True --image-size 224 --epochs 2 --lr 1e-3 --batch-size 16 --num-workers 2

python -m src.train_runner --model-name "unet_se_resnext101_32x4d" --encoder "se_resnext101_32x4d" --debug True --image-size 224 --epochs 2 --lr 1e-3 --batch-size 16 --num-workers 2 