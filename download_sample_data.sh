#!/bin/bash

cd actloc_benchmark/example_data

# download using gdown (install if missing)
if ! command -v gdown &> /dev/null
then
    echo "gdown not found, installing with pip..."
    pip install gdown
fi

# download sample data and unzip
gdown "https://drive.google.com/uc?id=1rNUKflwjPzpcHbb9Ssxdt51UKI9oy3_o" \
      -O example_data.zip
unzip -q example_data.zip
rm example_data.zip

cd ..