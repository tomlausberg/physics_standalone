#!/bin/bash

rundir=$1
scheme=$2

cd ./python/${rundir}
python test_${scheme}_gt4py.py
