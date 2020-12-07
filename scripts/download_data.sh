#!/bin/bash
"""
download
"""
CUR_DIR=$pwd
DATA_DIR_LOC=data

mkdir -p $DATA_DIR_LOC
cd $DATA_DIR_LOC

if [ "$(ls -A $(pwd))" ]
then
    echo "$(pwd) not empty!"
else
    echo "$(pwd) is empty!"
    kaggle competitions download -c nfl-impact-detection
    unzip nfl-impact-detection.zip .
    rm nfl-impact-detection.zip    
fi
cd $CUR_DIR
echo $(pwd)