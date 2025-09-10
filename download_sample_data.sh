#!/bin/bash
mkdir -p actloc_benchmark/example_data
cd actloc_benchmark/example_data

# download using gdown (install if missing)
if ! command -v gdown &> /dev/null
then
    echo "gdown not found, installing with pip..."
    pip install gdown
fi

# download sample data and unzip
gdown "https://drive.google.com/uc?id=1cL7FuSUetKux2LWOlD9sRpUi4NX1pRvr" \
      -O example_data.zip
unzip -q example_data.zip
rm example_data.zip

cd ..