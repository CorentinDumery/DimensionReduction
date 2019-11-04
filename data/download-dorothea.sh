#!/bin/sh
#
# This script downloads the DOROTHEA dataset

mkdir dorothea
cd dorothea

for i in {DOROTHEA/dorothea.param,DOROTHEA/dorothea_train.data,DOROTHEA/dorothea_train.labels,DOROTHEA/dorothea_valid.data,dorothea_valid.labels}; do
  echo "downloading $i..."
  curl -s -O http://archive.ics.uci.edu/ml/machine-learning-databases/dorothea/$i &> /dev/null;
done;

cat dorothea_train.labels dorothea_valid.labels > dorothea_all.labels
cat dorothea_train.data dorothea_valid.data     > dorothea_all.data

cd ..



exit 0
