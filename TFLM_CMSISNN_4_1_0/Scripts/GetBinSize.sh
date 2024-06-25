#! /usr/bin/bash
# usage: find ../tflite_models/ -name *.h.ori | xargs -i% GetBinSize.sh %

model_path=$1
model_filename=`echo "$model_path" | sed -n 's/.*\/\(.*\).h.ori/\1/p'`

cp $model_path model/model_data.cc
bin_path=`mbed compile --profile o3 | grep Image: | cut -d ' ' -f2`
bin_size=`du -b $bin_path | cut -d $'\t' -f1`

if [[ $bin_path == *".bin" ]]; then
    echo -e "${bin_size}\t${model_filename}"
else
    echo "Fault: ${model_filename}"
    set -e
fi


