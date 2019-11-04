#!/bin/sh
#
# This script downloads the DEXTER dataset

mkdir dexter
cd dexter

for i in {DEXTER/dexter.param,DEXTER/dexter_train.data,DEXTER/dexter_train.labels,DEXTER/dexter_valid.data,dexter_valid.labels}; do
  echo "downloading $i..."
  curl -s -O http://archive.ics.uci.edu/ml/machine-learning-databases/dexter/$i &> /dev/null;
done;

cat dexter_train.labels dexter_valid.labels > dexter_all.labels
cat dexter_train.data dexter_valid.data     > dexter_all.data

cd ..



exit 0
