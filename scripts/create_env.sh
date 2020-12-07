#!/bin/bash

conda update -y conda
conda create -y -n nfl python=3.7
conda activate nfl

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install --upgrade pip
pip install -r requirements.txt