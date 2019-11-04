#!/bin/sh
#
# This script downloads the GISETTE dataset

mkdir gisette
cd gisette

for i in {GISETTE/gisette.param,GISETTE/gisette_train.data,GISETTE/gisette_train.labels,GISETTE/gisette_valid.data,gisette_valid.labels}; do
  echo "downloading $i..."
  curl -s -O http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/$i &> /dev/null;
done;

cat gisette_train.labels gisette_valid.labels > gisette_all.labels
cat gisette_train.data gisette_valid.data     > gisette_all.data

cd ..



exit 0
