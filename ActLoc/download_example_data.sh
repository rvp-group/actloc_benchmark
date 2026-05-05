#!/bin/bash

# download using gdown (install if missing)
if ! command -v gdown &> /dev/null
then
    echo "gdown not found, installing with pip..."
    pip install gdown
fi

# download sample data and unzip
gdown https://drive.google.com/uc?id=1QssNeHx9-CzbloKmOe37NyQ0KfTRScwk \
      -O example_data.zip
unzip -q example_data.zip
rm example_data.zip

cd ..