#!/bin/bash

cd actloc_benchmark/example_data

# download using gdown (install if missing)
if ! command -v gdown &> /dev/null
then
    echo "gdown not found, installing with pip..."
    pip install gdown
fi

# download sample data and unzip
gdown "https://drive.google.com/uc?export=download&id=16GVRGFupL65CKmYK5qmjyY4YebP3y3Lo" \
      -O example_data.zip
unzip -q example_data.zip
rm example_data.zip

cd ..