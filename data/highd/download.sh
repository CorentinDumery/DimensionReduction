#!/bin/sh
#
# This script downloads the highd datasets

for i in {032,064,128,256,512,1024}; do
  echo "downloading $i..."
  curl -s -O https://cs.uef.fi/sipu/datasets/dim$i.txt &> /dev/null;
done;

curl -s -o gt.zip https://cs.uef.fi/sipu/datasets/dim-txt.zip &> /dev/null;

mkdir gt
yes | unzip -d gt gt.zip && rm gt.zip
