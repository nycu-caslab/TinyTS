#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdint>

#include "inc/tensorflow/quantization_util.h"

using namespace std;

// 0: executable path, 1: working directory
int main(int argc, char* argv[]){
    ifstream in;
    in.open(string(argv[1])+string("/input"), fstream::in);
    int data_in[3];
    for(int i=0; i<3; i++){
        in >> data_in[i];
    }
    in.close();

    float quant_scale_input1 = *reinterpret_cast<float*>(&data_in[0]);
    float quant_scale_input2 = *reinterpret_cast<float*>(&data_in[1]);
    float quant_scale_output = *reinterpret_cast<float*>(&data_in[2]);
    
    const double twice_max_input_scale =
        2 * static_cast<double>(
                std::max(quant_scale_input1, quant_scale_input2));
    const double real_input1_multiplier =
        static_cast<double>(quant_scale_input1) / twice_max_input_scale;
    const double real_input2_multiplier =
        static_cast<double>(quant_scale_input2) / twice_max_input_scale;
    const double real_output_multiplier =
        twice_max_input_scale /
        ((1 << 20/* left_shift */) * static_cast<double>(quant_scale_output));

    int32 input1_multiplier;
    int32 input2_multiplier;
    int32 output_multiplier;
    int input1_shift;
    int input2_shift;
    int output_shift;
    QuantizeMultiplierSmallerThanOneExp(
        real_input1_multiplier, &input1_multiplier, &input1_shift);

    QuantizeMultiplierSmallerThanOneExp(
        real_input2_multiplier, &input2_multiplier, &input2_shift);

    QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &output_multiplier, &output_shift);

    ofstream out;
    out.open(string(argv[1])+string("/output"), fstream::out);
    out << input1_multiplier << ' ' << input2_multiplier << ' ' << output_multiplier << endl;
    out << input1_shift << ' ' << input2_shift << ' ' << output_shift;
    out.close();
    return 0;

}