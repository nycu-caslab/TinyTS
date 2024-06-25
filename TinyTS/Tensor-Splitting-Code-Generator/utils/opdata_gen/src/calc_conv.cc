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
    int num_channels;
    in >> num_channels;
    int data_in[num_channels+2];
    for(int i=0; i<num_channels+2; i++){
        in >> data_in[i];
    }
    in.close();
    
    const float input_scale = reinterpret_cast<const float*>(&data_in[0])[0];
    const float output_scale = reinterpret_cast<const float*>(&data_in[1])[0];
    const float* filter_scales = reinterpret_cast<const float*>(&data_in[2]);
    
    int32_t per_channel_multiplier[num_channels];
    int32_t per_channel_shift[num_channels];
    
    for (int i = 0; i < num_channels; ++i) {
        // If per-tensor quantization parameter is specified, broadcast it along the
        // quantization dimension (channels_out).
        const float scale = /* is_per_channel ?  */filter_scales[i]/*  : filter_scales[0] */;
        const double filter_scale = static_cast<double>(scale);
        const double effective_output_scale = static_cast<double>(input_scale) *
                                            filter_scale /
                                            static_cast<double>(output_scale);
        int32_t significand;
        int channel_shift;
        QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
        per_channel_multiplier[i] = significand;
        per_channel_shift[i] = channel_shift;
    }

    ofstream out;
    out.open(string(argv[1])+string("/output"), fstream::out);
    for (int i = 0; i < num_channels; i++){
        out << per_channel_multiplier[i] << ' ';
    }
    out << endl;
    for (int i = 0; i < num_channels; i++){
        out << per_channel_shift[i] << ' ';
    }
    out.close();
    return 0;
}