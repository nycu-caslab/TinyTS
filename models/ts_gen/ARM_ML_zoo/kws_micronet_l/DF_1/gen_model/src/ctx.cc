#include "gen_model/include/ctx.h"
#include <cstdio>

alignas(16) int8_t arena[62208];
const int32_t arena_size = 62208;
const int32_t input_tid = 0;
const int32_t output_tid = 61;
const Tensor tensors[62] = {
    {{1, 49, 10, 1}, -1, 0, 0}, 
    {{276, 10, 4, 1}, 0, -1, 5}, 
    {{248, 1, 1, 276}, 11040, -1, 5}, 
    {{276, 1, 1, 248}, 79488, -1, 5}, 
    {{276, 1, 1, 276}, 147936, -1, 5}, 
    {{248, 1, 1, 276}, 224112, -1, 5}, 
    {{248, 1, 1, 248}, 292560, -1, 5}, 
    {{248, 1, 1, 248}, 354064, -1, 5}, 
    {{248, 1, 1, 248}, 415568, -1, 5}, 
    {{12, 1, 1, 248}, 477072, -1, 5}, 
    {{1, 276, 1, 1}, 480048, -1, -1}, 
    {{1, 248, 1, 1}, 481152, -1, -1}, 
    {{1, 276, 1, 1}, 482144, -1, -1}, 
    {{1, 276, 1, 1}, 483248, -1, -1}, 
    {{1, 248, 1, 1}, 484352, -1, -1}, 
    {{1, 248, 1, 1}, 485344, -1, -1}, 
    {{1, 248, 1, 1}, 486336, -1, -1}, 
    {{1, 248, 1, 1}, 487328, -1, -1}, 
    {{1, 12, 1, 1}, 488320, -1, -1}, 
    {{1, 2, 1, 1}, 488368, -1, 17}, 
    {{1, 2, 1, 1}, 488384, -1, 17}, 
    {{1, 2, 1, 1}, 488400, -1, 17}, 
    {{1, 2, 1, 1}, 488416, -1, 17}, 
    {{1, 2, 1, 1}, 488432, -1, 17}, 
    {{1, 2, 1, 1}, 488448, -1, 17}, 
    {{1, 1, 1, 1}, 488464, -1, 17}, 
    {{1, 49, 10, 1}, -49, 0, 17}, 
    {{1, 49, 10, 276}, -49, 49, 18}, 
    {{1, 3, 3, 276}, 488480, -1, 5}, 
    {{1, 276, 1, 1}, 490976, -1, -1}, 
    {{1, 2, 1, 1}, 492080, -1, 17}, 
    {{1, 25, 5, 276}, -25, 98, 19}, 
    {{1, 25, 5, 248}, -25, 123, 20}, 
    {{1, 3, 3, 248}, 492096, -1, 5}, 
    {{1, 248, 1, 1}, 494336, -1, -1}, 
    {{1, 25, 5, 248}, -25, 148, 21}, 
    {{1, 25, 5, 276}, -25, 173, 22}, 
    {{1, 3, 3, 276}, 495328, -1, 5}, 
    {{1, 276, 1, 1}, 497824, -1, -1}, 
    {{1, 25, 5, 276}, -25, 198, 23}, 
    {{1, 25, 5, 276}, -25, 223, 24}, 
    {{1, 3, 3, 276}, 498928, -1, 5}, 
    {{1, 276, 1, 1}, 501424, -1, -1}, 
    {{1, 25, 5, 276}, -25, 248, 25}, 
    {{1, 25, 5, 248}, -25, 273, 26}, 
    {{1, 3, 3, 248}, 502528, -1, 5}, 
    {{1, 248, 1, 1}, 504768, -1, -1}, 
    {{1, 25, 5, 248}, -25, 298, 27}, 
    {{1, 25, 5, 248}, -25, 323, 28}, 
    {{1, 3, 3, 248}, 505760, -1, 5}, 
    {{1, 248, 1, 1}, 508000, -1, -1}, 
    {{1, 25, 5, 248}, -25, 348, 29}, 
    {{1, 25, 5, 248}, -25, 373, 30}, 
    {{1, 3, 3, 248}, 508992, -1, 5}, 
    {{1, 248, 1, 1}, 511232, -1, -1}, 
    {{1, 25, 5, 248}, -25, 398, 31}, 
    {{1, 25, 5, 248}, -25, 423, 32}, 
    {{1, 25, 5, 248}, -1, 0, 1}, 
    {{1, 1, 1, 248}, -1, 31008, 2}, 
    {{1, 1, 1, 12}, -1, 0, 3}, 
    {{1, 2, 1, 1}, 512224, -1, 17}, 
    {{1, 12, 1, 1}, -1, 16, 4}
};
const int all_0_zp_cursor = 5;
const int only_zp_cursor = 17;
const int only_zp_start = 281;
const int32_t quant_scale[17] = {
    1045294039, 1019265217, 1019265217, 1028739976, 1028739976, 987024247, 986037565, 985650945, 986740357, 985834462, 987038115, 986290393, 986045776, 986139859, 985781031, 985144974, 989977471
};
const int8_t quant_zeropoint[297] = {
    -5, -128, -128, -63, -63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128
};
const int32_t split_offset[448] = {
    9696, 9712, 12480, 12496, 15120, 15136, 21504, 21520, 24000, 24016, 25248, 25264, 26496, 26512, 26528, 26544, 26560, 26576, 26592, 26608, 26624, 26640, 26656, 26672, 26688, 26704, 26720, 26736, 26752, 26768, 26784, 26800, 26816, 26832, 26848, 26864, 26880, 26896, 26912, 26928, 26944, 26960, 26976, 26992, 27008, 27024, 27040, 27056, 27072, 0, 2768, 0, 5536, 0, 2768, 0, 5536, 0, 2768, 0, 5536, 0, 2768, 0, 5536, 0, 2768, 0, 5536, 0, 2768, 0, 5536, 0, 2768, 0, 5536, 0, 2768, 0, 5536, 0, 2768, 0, 5536, 0, 2768, 0, 5536, 0, 2768, 0, 5536, 0, 2768, 0, 5536, 0, 5536, 8304, 9696, 12480, 13872, 11088, 8304, 9696, 12480, 13872, 11088, 8304, 9696, 12480, 13872, 11088, 8304, 9696, 12480, 13872, 11088, 8304, 9696, 12480, 12480, 11088, 13872, 15264, 16512, 19008, 21504, 24000, 25248, 22752, 20256, 17760, 15264, 16512, 19008, 21504, 24000, 25248, 22752, 20256, 17760, 15264, 16512, 19008, 21504, 1392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4160, 6944, 8304, 9696, 12480, 13872, 11088, 8304, 9696, 12480, 13872, 11088, 8304, 9696, 12480, 13872, 11088, 8304, 9696, 12480, 13872, 11088, 8304, 9696, 2768, 0, 4160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4160, 9696, 6944, 11088, 8304, 9696, 12480, 13872, 11088, 8304, 9696, 12480, 13872, 11088, 8304, 9696, 12480, 13872, 11088, 8304, 9696, 12480, 13872, 11088, 8304, 5552, 1392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4160, 9696, 2784, 2784, 17760, 15264, 16512, 19008, 21504, 24000, 25248, 22752, 20256, 17760, 15264, 16512, 19008, 21504, 24000, 25248, 22752, 20256, 17760, 15264, 13872, 11088, 8336, 4176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4160, 2784, 2784, 1248, 1248, 20256, 17760, 15264, 16512, 19008, 21504, 24000, 25248, 22752, 20256, 17760, 15264, 16512, 19008, 21504, 24000, 25248, 22752, 20256, 16512, 12336, 9584, 5424, 2496, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4160, 2784, 2784, 1248, 1248, 1248, 22752, 20256, 17760, 15264, 16512, 19008, 21504, 24000, 25248, 22752, 20256, 17760, 15264, 16512, 19008, 21504, 24000, 25248, 17760, 15120, 13584, 6672, 7920, 3744, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4160, 2784, 2784, 1248, 1248, 1248, 1248, 31008, 32256, 33504, 34752, 36000, 37248, 38496, 39744, 40992, 42240, 43488, 44736, 45984, 47232, 48480, 49728, 50976, 52224, 53472, 54720, 55968, 57216, 58464, 59712, 60960
};
alignas(16) const uint8_t offline_tensor_data[512240] = {
};

int CtxSummary(){
    printf("Arena Size: %d\n", arena_size);
    printf("Tensor Metadata Summary:\n");
    printf("\ttensors: %d\n",sizeof(tensors));
    printf("\tquant_scale: %d\n",15684);
    printf("\tquant_zeropoint: %d\n",297);
    printf("\tsplit_offset: %d\n",sizeof(split_offset));
    printf("\toffline_tensor_data: %d\n",sizeof(offline_tensor_data));

    int byte_tensor =   sizeof(tensors) +
                        15684 + 297 +
                        sizeof(split_offset) + sizeof(offline_tensor_data);
    return byte_tensor;
}