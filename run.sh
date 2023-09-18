#!/bin/bash
###################################################################
# Script Name : run.sh
# Description : Trains and tests all the ML models
# Args        : -
# Author      : Billy Lim Jun Ming
# Email       : billy.ljm@gmail.com
###################################################################

python src/model_dummy.py
python src/model_svm.py
python src/model_forest.py
python src/model_boost.py