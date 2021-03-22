#!/bin/bash

# check if data is downloaded use wget
# train data links
TRAINIMG="https://s3.eu-central-1.amazonaws.com/aicrowd-static/datasets/snake-species-identification-challenge/train.tar.gz"

# test data links
TESTIMG="https://s3.eu-central-1.amazonaws.com/aicrowd-static/datasets/snake-species-identification-challenge/round1_test.tar.gz"

# download directory
DOWNLOADDIR=../teams/DSC180B_WI21_A00/a01capstonegroup06/

while true; do
    read -p "Do you wish to install this program (approximately 25 GB for downloading the data)?" yn
    case $yn in
        [Yy]* ) 
            read -p "Installing to directory $DOWNLOADDIR, would you like to change directory?" yn
            case $yn in 
                [Yy]* )
                    echo "Enter new directory to download to: "
                    read NEWDOWNLOADDIR
                    echo "Downloading to $NEWDOWNLOADDIR"                   
                    
                    echo "Downloading train_images.tar.gz from $TRAINIMG..."
                    wget -nc -P $NEWDOWNLOADDIR $TRAINIMG
                    tar zxvf "${NEWDOWNLOADDIR}train.tar.gz" -C $NEWDOWNLOADDIR
                    
                    echo "Downloading validate_images.tar.gz from $TESTIMG..."
                    wget -nc -P $NEWDOWNLOADDIR $TESTIMG
                    tar zxvf "${NEWDOWNLOADDIR}round1_test.tar.gz" -C $NEWDOWNLOADDIR
                    
                    exit
                    ;;
                [Nn]* )
                    # downloading data
                    echo "Downloading train_images.tar.gz from $TRAINIMG..."
                    wget -nc -P $DOWNLOADDIR $TRAINIMG
                    tar zxvf "${DOWNLOADDIR}train.tar.gz" -C $DOWNLOADDIR
                    
                    echo "Downloading validate_images.tar.gz from $TESTIMG..."
                    wget -nc -P $DOWNLOADDIR $TESTIMG
                    tar zxvf "${DOWNLOADDIR}round1_test.tar.gz" -C $DOWNLOADDIR

                    exit
                    ;;
                * ) 
                echo "Please answer yes or no."
                ;;
            esac
            ;;
        [Nn]* ) 
        exit
        ;;
        * ) 
        echo "Please answer yes or no."
        ;;
    esac
    

done
