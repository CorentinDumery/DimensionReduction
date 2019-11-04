#!/bin/sh
#
# This script downloads the taxis datasets

echo "Downloading taxis dataset..."
curl -s -O https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-01.csv &> /dev/null;

exit 0
