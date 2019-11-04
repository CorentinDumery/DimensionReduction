#!/bin/sh
#
# This script downloads the MNIST dataset
# Source: https://www.python-course.eu/neural_network_mnist.php

mkdir mnist
cd mnist

for i in {train,test}; do
  echo "downloading $i..."
  curl -s -O https://www.python-course.eu/data/mnist/mnist_$i.csv &> /dev/null;
done;

cat mnist_test.csv mnist_train.csv > mnist_all.csv

cd ..



exit 0
