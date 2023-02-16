#!/bin/bash

##Exit Status Function
function exit_status () {
if [ $? -eq 0 ]
    then
        echo -e "\n [INFO] Success: $1"
    else
        echo -e "\n [ERROR] Errors detected: $1"
    exit 1
fi
}

function log_info() {
  echo "========================================"
  echo " [INFO] $1"
  echo "========================================"
}

function download_extract() {
    cd "$DATA_DIR" || exit_status "Changing Directory $FRAMES_DIR"
    echo "Downloading data for: $link"
    wget "$link"
    cd "$FRAMES_DIR" || exit_status "Changing Directory $DATA_DIR"
    zip_file=$(echo $link | awk -F/ '{print $NF}')
    unzip "$DATA_DIR/$zip_file"
    # echo $(echo $zip_file | awk '{split($0,a,"."); print a[1]}')
    folder_name=${zip_file%.*}
    cd "$folder_name" || exit_status "Changing Directory $folder_name"
    mv * ../
    cd "$FRAMES_DIR" || exit_status "Changing Directory $FRAMES_DIR"
    rm -rf "$folder_name"
    # rm ${cam1_links[$i]##*/}
}


function download_cam0_rgb_datatset() {
    log_info "Downloading cam0 rgb dataset"
    declare -a cam0_links=(
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-01-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-02-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-03-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-04-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-05-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-06-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-07-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-08-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-09-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-10-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-11-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-12-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-13-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-14-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-15-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-16-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-17-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-18-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-19-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-20-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-21-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-22-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-23-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-24-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-25-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-26-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-27-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-28-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-29-cam0-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-30-cam0-rgb.zip
    )
    length=${#cam0_links[@]}
    echo "Total number of links: $length"
    for (( i=0; i<${length}; i++ )); do
        link=${cam0_links[$i]}
        echo "Downloading $link"
        download_extract
    done
    printf "\n"
}

function download_cam1_rgb_datatset() {
    log_info "Downloading cam1 rgb dataset"
    declare -a cam1_links=(
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-01-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-02-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-03-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-04-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-05-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-06-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-07-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-08-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-09-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-10-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-11-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-12-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-13-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-14-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-15-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-16-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-17-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-18-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-19-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-20-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-21-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-22-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-23-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-24-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-25-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-26-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-27-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-28-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-29-cam1-rgb.zip
        http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-30-cam1-rgb.zip
    )
    length=${#cam1_links[@]}
    for (( i=0; i<${length}; i++ )); do
        link=${cam1_links[$i]}
        download_extract
    done
    printf "\n"
}

function  download_csv() {
    cd $BASE_DIR || exit_status "Cannot change directory to $CSV_DIR"
    CSV_LINK="http://fenix.univ.rzeszow.pl/~mkepski/ds/data/urfall-cam0-falls.csv"
    echo "Downloading CSV file from $CSV_LINK"
    wget $CSV_LINK
    echo "Adding headers to CSV file"
    echo "sequenceName,frameNumber,label,HeightWidthRatio,MajorMinorRatio,BoundingBoxOccupancy,MaxStdXZ,HHmaxRatio,H,D,P40" > /tmp/headers
    cat /tmp/headers urfall-cam0-falls.csv > /tmp/urfall-cam0-falls.csv
    mv /tmp/urfall-cam0-falls.csv .
    printf "\n"
}

WORK_DIR=$(pwd)
#WORK_DIR=/run/media/danish/404/Mot/UR
BASE_DIR="$WORK_DIR/datasets/UR"
FRAMES_DIR="$BASE_DIR/Frames"
DATA_DIR="$BASE_DIR/data"
echo "Working Directory: $WORK_DIR"
echo "Frames Directory: $FRAMES_DIR"
echo "Data Directory: $DATA_DIR"
echo "Creating directory $FRAMES_DIR"
mkdir -p $FRAMES_DIR
mkdir -p $DATA_DIR

download_csv
download_cam0_rgb_datatset
download_cam1_rgb_datatset