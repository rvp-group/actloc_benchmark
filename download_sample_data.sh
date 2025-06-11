#!/bin/bash

cd actloc_benchmark

# download using gdown (install if missing)
if ! command -v gdown &> /dev/null
then
    echo "gdown not found, installing with pip..."
    pip install gdown
fi

# download sample data and unzip
gdown "https://drive.google.com/uc?id=1ZLEts4G_nTbM6VXq-iwSXtc-ISw92_xH"
unzip -q example_data.zip
rm example_data.zip