#!/bin/bash

#directory containing google cloud bucket data
export DATA_DIR=$1
 

if [ ! -z "$IS_DOCKER" ] ; then
    echo "This script cannot be run in the Docker image"
else

    export MYHOME=`pwd`

    mkdir -p fortran
    cd ./fortran
    mkdir -p data
    cd data
    mkdir -p LW
    mkdir -p SW
    cd ..
    mkdir -p radlw
    mkdir -p radsw

    cd $MYHOME

    cp -r $DATA_DIR/fv3gfs-fortran-output/lwrad/* ./fortran/data/LW/.
    cd ./fortran/data/LW
    tar -xzvf data.tar.gz
    cd $MYHOME   

    cp -r $DATA_DIR/fv3gfs-fortran-output/swrad/* ./fortran/data/SW/.
    cd ./fortran/data/SW
    tar -xzvf data.tar.gz
    cd $MYHOME

    mkdir -p python
    cd ./python
    mkdir -p lookupdata
    cd $MYHOME

    cp -r $DATA_DIR/lookupdata/lookup.tar.gz ./python/lookupdata/.
    cd ./python/lookupdata
    tar -xzvf lookup.tar.gz
    cd $MYHOME

    cd ./fortran/radlw
    mkdir -p dump
    cd $MYHOME

    cd ./fortran/radsw
    mkdir -p dump
    cd $MYHOME

    cp -r $DATA_DIR/standalone-output/lwrad/* ./fortran/radlw/dump/.
    cd ./fortran/radlw/dump
    tar -xzvf data.tar.gz
    cd $MYHOME

    cp $DATA_DIR/standalone-output/swrad/* ./fortran/radsw/dump/.
    cd ./fortran/radsw/dump
    tar -xzvf data.tar.gz
    cd $MYHOME

    cd ./python
    mkdir -p forcing
    cd $MYHOME

    cp $DATA_DIR/forcing/* ./python/forcing/.
    cd ./python/forcing
    tar -xzvf data.tar.gz
    cd $MYHOME
    
    cd ./fortran
    mkdir -p radlw
    cd ./radlw
    cp -r $DATA_DIR/fortran_data .
fi
