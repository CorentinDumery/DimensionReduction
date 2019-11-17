#!/bin/sh
#
# This script downloads all of the required datasets

mkdir data
cd data

sh download-mnist.sh
# sh download-dexter.sh
# sh download-dorothea.sh
sh download-gisette.sh
# sh download-taxis.sh
sh download-highd.sh

cd ..

exit 0
