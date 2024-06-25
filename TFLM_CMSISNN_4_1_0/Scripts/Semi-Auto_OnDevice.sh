#! /usr/bin/bash
# usage: find ../tflite_models/ -name *.h.ori | xargs -i% Semi-Auto_OnDevice.sh %

# read model_path
compile_cmd='mbed compile --profile o3 | grep Image: | cut -d '"' '"' -f2'
flash_cmd='STM32_Programmer_CLI.exe -c port=SWD freq=4000 ap=0 -d ${bin_path} 0x08000000 -v'
terminal_cmd='mbed sterm -b 115200 -r'

model_path=$1
if [[ $model_path != *".h.ori" ]]; then
    echo "Fault: ${model_path}"
    exit 1
fi

model_filename=`echo "$model_path" | sed -n 's/.*\/\(.*\).h.ori/\1/p'`
echo -n "Skip ${model_filename}? (y/N):"
read ans
if [[ $ans != *"y"* ]]; then
    echo "Start: ${model_filename}"

    cp $model_path model/model_data.cc
    echo "Compile: ${model_filename}"
    bin_path=$(eval ${compile_cmd})
    echo "Compile done"
    bin_size=`du -b $bin_path | cut -d $'\t' -f1`

    echo "Flash: ${model_filename}"
    eval ${flash_cmd}
    echo "Flash done"

    eval ${terminal_cmd}

    echo $model_filename
    while true
    do
        brk=true
        echo -n 'Redo?(y/N): '
        read ans
        # redo compile
        if [[ $ans == *"c"* ]]; then
            bin_path=$(eval ${compile_cmd})
            brk=false
        fi
        # redo flash
        if [[ $ans == *"f"* ]]; then
            eval ${flash_cmd}
            brk=false
        fi
        # redo term
        if [[ $ans == *"t"* ]]; then
            eval ${terminal_cmd}
            brk=false
        fi
        if $brk ; then
            break
        fi
    done
else
    echo "Skip: ${model_filename} ${bin_path}"
fi

