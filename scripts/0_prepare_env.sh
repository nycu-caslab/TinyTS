#! /usr/bin/bash

# Prepare OpData Generator
old_pwd=$(pwd)
repo_root_path=$(readlink -f $(dirname $(readlink -f $0))/..)
cd ${repo_root_path}/TinyTS/Tensor-Splitting-Code-Generator/utils/opdata_gen
mkdir -p build
cd build
rm *
cmake ..
make
cd ${old_pwd}

# Prepare models
cd ${repo_root_path}
python models/prepare_tflite_models.py
python models/prepare_tflm_models.py
python models/prepare_ts_models.py
python models/prepare_ts_gen.py
python models/prepare_ts_gen.py --evict_in
cd ${old_pwd}

# Build docker image
cd ${repo_root_path}
cd docker/
docker build -t=tinyts .
cd ${old_pwd}