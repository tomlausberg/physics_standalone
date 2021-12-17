#!/bin/bash

export DATA_DIR=$1
 

if [ ! -z "$IS_DOCKER" ] ; then
    echo "This script cannot be run in the Docker image"
else

    export MYHOME=`pwd`

    if [ ! -d "./fortran/data" ]; then
        cd ./fortran
        mkdir data
        cd data
        mkdir LW
        mkdir SW
        cd $MYHOME
    else
        echo "Fortran output directory already exists"
    fi

    if [ -z "$(ls -A ./fortran/data/LW)" ]; then
        cp -r $DATA_DIR/fv3gfs-fortran-output/lwrad/* ./fortran/data/LW/.
        cd ./fortran/data/LW
        tar -xzvf data.tar.gz
        cd $MYHOME
    else
        echo "LW Fortran data already present"
    fi

    if [ -z "$(ls -A ./fortran/data/SW)" ]; then
        cp -r $DATA_DIR/fv3gfs-fortran-output/swrad/* ./fortran/data/SW/.
        cd ./fortran/data/SW
        tar -xzvf data.tar.gz
        cd $MYHOME
    else
        echo "SW Fortran data already present"
    fi

    cd ./python
    mkdir lookupdata
    cd $MYHOME

    if [ -z "$(ls -A ./python/lookupdata)" ]; then
        cp -r $DATA_DIR/lookupdata/lookup.tar.gz ./python/lookupdata/.
        cd ./python/lookupdata
        tar -xzvf lookup.tar.gz
        cd $MYHOME
    else
        echo "Data already present"
    fi

    if [ ! -d "./fortran/radlw/dump" ]; then
        cd ./fortran/radlw
        mkdir dump
        cd $MYHOME
    else
        echo "LW standalone output directory already exists"
    fi

    if [ ! -d "./fortran/radsw/dump" ]; then
        cd ./fortran/radsw
        mkdir dump
        cd $MYHOME
    else
        echo "SW standalone output directory already exists"
    fi

    if [ -z "$(ls -A ./fortran/radlw/dump)" ]; then
        cp $DATA_DIR/standalone-output/lwrad/* ./fortran/radlw/dump/.
        cd ./fortran/radlw/dump
        tar -xzvf data.tar.gz
        cd $MYHOME
    else
        echo "LW standalone data already present"
    fi

    if [ -z "$(ls -A ./fortran/radsw/dump)" ]; then
        cp $DATA_DIR/standalone-output/swrad/* ./fortran/radsw/dump/.
        cd ./fortran/radsw/dump
        tar -xzvf data.tar.gz
        cd $MYHOME
    else
        echo "SW standalone data already present"
    fi

    if [ ! -d "./python/forcing" ]; then
        cd ./python
        mkdir forcing
        cd $MYHOME
    else
        echo "Forcing directory already exists"
    fi

    if [ -z "$(ls -A ./python/forcing)" ]; then
	    cp $DATA_DIR/forcing/* ./python/forcing/.
	    cd ./python/forcing
	    tar -xzvf data.tar.gz
        cd $MYHOME
    else
	    echo "Forcing data already present"
    fi  
fi
