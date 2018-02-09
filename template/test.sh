#!/bin/bash

make all
cd build/bin &&
./task2 -d ../../data/multiclass/train_labels.txt -m model.txt --train &&
./task2 -d ../../data/multiclass/train_labels.txt -m model.txt -l predictions.txt --predict&&
cd ../../ &&
./compare.py data/multiclass/train_labels.txt build/bin/predictions.txt

exit 0