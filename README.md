# TinyTS: Memory-Efficient TinyML Model Compiler Framework on Microcontrollers
Yu-Yuan Liu, Hong-Sheng Zheng, Yu-Fang Hu, Chen-Fong Hsu, Tsung Tai Yeh

The 30th IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2024

\[[paper](https://ieeexplore.ieee.org/abstract/document/10476479/)\]

# Introduction
Deploying DNN model onto microcontrollers is typically limited by the tight SRAM budget. TinyTS proposed Tensor-Spliting method, which modifies the computational graph of a DNN model (TinyTS targets on CNN) to process one small chopped tensor at a time. So that TinyTS can run DNN model inference on microcontrollers with extremely small arena space, almost the same inference latency (~+5%) and no accuracy loss. Comparing to prior works, TinyTS have 5.92X memory-efficiency vs. [TFLM](https://github.com/tensorflow/tflite-micro), and runs inference 8.83x faster than [MCUNetV2](https://github.com/mit-han-lab/tinyengine)'s patch-based inference when using equaly small memory space across 9 TinyML model from [MCUNet model zoo](https://github.com/mit-han-lab/mcunet/blob/24dbe23c83ad876a80a331dd2f112b4d096413ff/mcunet/model_zoo.py).

# Third-party code
Some parts of code of TinyTS are based on prior works. This section explains the relationship between each module and the related prior work.

## Benchmark Suite
- Description:
    - To run inference and get measurements.
- Related Third-party project:
    - [MLPerf Tiny](https://github.com/mlcommons/tiny)
- Corresponding directories and files:
    - TinyTS/runtime
        - api/
        - util/
        - submitter_implemented.cc
        - main.cpp
## TFlite flatbuffers schema
- Description:
    - We use `flatc` to convert .tflite to .json to modify the tflite model. To prevent precision loss, all the float type are changed to int type.
- Related Third-party project:
    - [Tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/schema)
- Corresponding directories and files:
    - TinyTS/Tensor-Splitting-Code-Generator/utils/schema.fbs
## Code Generator
- Description:
    - For generating the code sequence to call TinyEngine's kernels.
- Related Third-party project:
    - [TinyEngine](https://github.com/mit-han-lab/tinyengine)
- Corresponding directories and files:
    - TinyTS/Tensor-Splitting-Code-Generator/
        - TE_codegen_template/
        - code_gen.py
        - depthwiseTemplate.py
## OP Data Generator (Requantization Parameter Generator)
- Description:
    - For generating requatization parameter of CMSIS-NN kernel.
- Related Third-party project:
    - [TFLM](https://github.com/tensorflow/tflite-micro)
- Corresponding directories and files:
    - TinyTS/Tensor-Splitting-Code-Generator/utils/opdata_gen/
## Operator API
- Description:
    - For processing the parameters of CMSIS-NN kernel.
- Related Third-party project:
    - [TFLM](https://github.com/tensorflow/tflite-micro)
- Corresponding directories and files:
    - TinyTS/runtime/gen_lib/
## Kernel Library
- Description:
    - TinyTS extends the following kernel library to make it able to execute on a fragmented tensor to make them adapt to TinyTS's VFP.
- Related Third-party project:
    - [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)
    - [TinyEngine](https://github.com/mit-han-lab/tinyengine)
- Corresponding directories and files:
    - TinyTS/runtime/cmsis/
    - TinyTS/runtime/cmsis_nn_deprecated/
    - TinyTS/runtime/third-party/
    - TinyTS/runtime/TinyEngine/
    - The `codegen/` dir created by code generator
## Reference Baseline
- Description:
    - A memory-efficiency baseline extended from MLPerf Tiny.
- Related Third-party project:
    - [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)
    - [MLPerf Tiny](https://github.com/mlcommons/tiny)
    - [TFLM](https://github.com/tensorflow/tflite-micro)
- Corresponding directories and files:
    - TFLM_CMSISNN_4_1_0/

# Requirements
We provide a docker image to run our framework. Please install docker first before using TinyTS. 

# Environment Setup
## Prepare
Please run the following command to compile Opdata generator, download tflite models and setup docker image.
```
bash scripts/0_prepare_env.sh
```
## Run
The following command run a container for runing TinyTS's scripts and using mbed tool to compile program, flash and open serial session to dev boards.
```
bash docker_run.sh
```

# Usage
## Tensor-Splitting Model
We suggest using the scripts under `models/` directory to get your Tensor-Spling model and code-gen C model, please see the [README.md here](models/tflite/MLPerf/README.md).

### You can refer to the following files for the usage of Graph-Rewriter and Code-Genertor
- Graph-Rewriter
    - models/prepare_ts_models.py
    - TinyTS/Tensor-Splitting-Graph-Rewriter/Compiler.py
- Code-Generator
    - models/prepare_ts_gen.py
    - TinyTS/Tensor-Splitting-Code-Generator/main.py

## Running inference on microcontrollers
### Requirements
Since we conducted the experiment with NEUCLEO-F767ZI, the environment provided below is for mbed-tools toolchain. It may not compatible for other platforms.

### Getting started
Before compiling a mbed program, you should make sure the mbed-os/ folder presents. If not, please run the following command to check-out mbed-os repo.
```
cd <TFLM_CMSISNN_4_1_0 | TinyTS/runtime>
mbed deploy
```

After check-out mbed-os repo, if you want to build your program with O3 optimization level, please duplicate release build profile and rename to `o3.json`. Then, replace `-Os` flag with `-O3` fag. Finally, you can build your program with your new build profile with O3 optimization level.
```
cd <TFLM_CMSISNN_4_1_0 | TinyTS/runtime>
cp mbed-os/tools/profiles/release.json mbed-os/tools/profiles/o3.json
sed -i -e 's/\-Os/-O3/g' mbed-os/tools/profiles/o3.json
mbed compile --profile o3
```

### Reference Baseline - TFLM
Go into TFLM direcory, replace model_data.cc, compile program, flash binary, and open terminal.

Here we takes MCUNet ImageNet 5fps model as example.
```
cd TFLM_CMSISNN_4_1_0
cp -r ../models/tflm_format/MCUNet_model_zoo/3_IM5/model_data.cc model/model_data.cc
mbed compile -f --sterm --baud 115200
```

### TinyTS
Go into TinyTS runtime direcory, replace codegen/ and gen_model/, compile program, flash binary, and open terminal.

Here we takes MCUNet ImageNet 5fps model with DF execution order, split_height of 2 and evict_in turn on as example.
```
cd TinyTS/runtime
cp -r ../../models/ts_gen_evict_in/MCUNet_model_zoo/3_IM5/DF_2/* .
mbed compile -f --sterm --baud 115200
```
# Citation
If you use TinyTS in your research, please cite our paper in HPCA. Thank you!
```
@INPROCEEDINGS{10476479,
  author={Liu, Yu-Yuan and Zheng, Hong-Sheng and Fang Hu, Yu and Hsu, Chen-Fong and Yeh, Tsung Tai},
  booktitle={2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA)}, 
  title={TinyTS: Memory-Efficient TinyML Model Compiler Framework on Microcontrollers}, 
  year={2024},
  volume={},
  number={},
  pages={848-860},
  keywords={Schedules;Tensors;Runtime;Microcontrollers;Computational modeling;Source coding;Random access memory;TinyML;Deep Neural Network;Compiler;AIoT},
  doi={10.1109/HPCA57654.2024.00070}}
```
