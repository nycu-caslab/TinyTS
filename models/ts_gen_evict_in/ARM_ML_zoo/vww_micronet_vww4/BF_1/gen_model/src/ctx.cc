#include "gen_model/include/ctx.h"
#include <cstdio>

alignas(16) int8_t arena[45056];
const int32_t arena_size = 45056;
const int32_t input_tid = 0;
const int32_t output_tid = 193;
int8_t *model_input_data;
const Tensor tensors[194] = {
    {{1, 128, 128, 1}, -1, 0, 0}, 
    {{8, 3, 3, 1}, 0, -1, 5}, 
    {{4, 1, 1, 8}, 80, -1, 5}, 
    {{4, 1, 1, 4}, 112, -1, 5}, 
    {{8, 1, 1, 4}, 128, -1, 5}, 
    {{12, 1, 1, 8}, 160, -1, 5}, 
    {{14, 1, 1, 12}, 256, -1, 5}, 
    {{16, 1, 1, 14}, 432, -1, 5}, 
    {{28, 1, 1, 16}, 656, -1, 5}, 
    {{32, 1, 1, 28}, 1104, -1, 5}, 
    {{96, 1, 1, 32}, 2000, -1, 5}, 
    {{28, 1, 1, 96}, 5072, -1, 5}, 
    {{96, 1, 1, 32}, 7760, -1, 5}, 
    {{28, 1, 1, 96}, 10832, -1, 5}, 
    {{152, 1, 1, 32}, 13520, -1, 5}, 
    {{44, 1, 1, 152}, 18384, -1, 5}, 
    {{268, 1, 1, 44}, 25072, -1, 5}, 
    {{40, 1, 1, 268}, 36864, -1, 5}, 
    {{268, 1, 1, 44}, 47584, -1, 5}, 
    {{52, 1, 1, 268}, 59376, -1, 5}, 
    {{268, 1, 1, 52}, 73312, -1, 5}, 
    {{44, 1, 1, 268}, 87248, -1, 5}, 
    {{192, 1, 1, 52}, 99040, -1, 5}, 
    {{68, 1, 1, 192}, 109024, -1, 5}, 
    {{172, 1, 1, 68}, 122080, -1, 5}, 
    {{68, 1, 1, 172}, 133776, -1, 5}, 
    {{172, 1, 1, 68}, 145472, -1, 5}, 
    {{48, 1, 1, 172}, 157168, -1, 5}, 
    {{288, 1, 1, 68}, 165424, -1, 5}, 
    {{16, 1, 1, 288}, 185008, -1, 5}, 
    {{96, 1, 1, 16}, 189616, -1, 5}, 
    {{16, 1, 1, 96}, 191152, -1, 5}, 
    {{96, 1, 1, 16}, 192688, -1, 5}, 
    {{32, 1, 1, 96}, 194224, -1, 5}, 
    {{96, 1, 1, 32}, 197296, -1, 5}, 
    {{32, 1, 1, 96}, 200368, -1, 5}, 
    {{128, 1, 1, 32}, 203440, -1, 5}, 
    {{2, 1, 1, 128}, 207536, -1, 5}, 
    {{1, 8, 1, 1}, 207792, -1, -1}, 
    {{1, 4, 1, 1}, 207824, -1, -1}, 
    {{1, 4, 1, 1}, 207840, -1, -1}, 
    {{1, 8, 1, 1}, 207856, -1, -1}, 
    {{1, 12, 1, 1}, 207888, -1, -1}, 
    {{1, 14, 1, 1}, 207936, -1, -1}, 
    {{1, 16, 1, 1}, 208000, -1, -1}, 
    {{1, 28, 1, 1}, 208064, -1, -1}, 
    {{1, 32, 1, 1}, 208176, -1, -1}, 
    {{1, 96, 1, 1}, 208304, -1, -1}, 
    {{1, 28, 1, 1}, 208688, -1, -1}, 
    {{1, 96, 1, 1}, 208800, -1, -1}, 
    {{1, 28, 1, 1}, 209184, -1, -1}, 
    {{1, 152, 1, 1}, 209296, -1, -1}, 
    {{1, 44, 1, 1}, 209904, -1, -1}, 
    {{1, 268, 1, 1}, 210080, -1, -1}, 
    {{1, 40, 1, 1}, 211152, -1, -1}, 
    {{1, 268, 1, 1}, 211312, -1, -1}, 
    {{1, 52, 1, 1}, 212384, -1, -1}, 
    {{1, 268, 1, 1}, 212592, -1, -1}, 
    {{1, 44, 1, 1}, 213664, -1, -1}, 
    {{1, 192, 1, 1}, 213840, -1, -1}, 
    {{1, 68, 1, 1}, 214608, -1, -1}, 
    {{1, 172, 1, 1}, 214880, -1, -1}, 
    {{1, 68, 1, 1}, 215568, -1, -1}, 
    {{1, 172, 1, 1}, 215840, -1, -1}, 
    {{1, 48, 1, 1}, 216528, -1, -1}, 
    {{1, 288, 1, 1}, 216720, -1, -1}, 
    {{1, 16, 1, 1}, 217872, -1, -1}, 
    {{1, 96, 1, 1}, 217936, -1, -1}, 
    {{1, 16, 1, 1}, 218320, -1, -1}, 
    {{1, 96, 1, 1}, 218384, -1, -1}, 
    {{1, 32, 1, 1}, 218768, -1, -1}, 
    {{1, 96, 1, 1}, 218896, -1, -1}, 
    {{1, 32, 1, 1}, 219280, -1, -1}, 
    {{1, 128, 1, 1}, 219408, -1, -1}, 
    {{1, 2, 1, 1}, 219920, -1, -1}, 
    {{1, 2, 1, 1}, 219936, -1, 7}, 
    {{1, 1, 1, 1}, 219952, -1, 7}, 
    {{1, 128, 128, 1}, -128, 0, 7}, 
    {{1, 64, 64, 8}, -64, 128, 8}, 
    {{1, 64, 64, 4}, -64, 192, 9}, 
    {{1, 3, 3, 4}, 219968, -1, 5}, 
    {{1, 4, 1, 1}, 220016, -1, -1}, 
    {{1, 2, 1, 1}, 220032, -1, 7}, 
    {{1, 64, 64, 4}, -64, 256, 10}, 
    {{1, 2, 1, 1}, 220048, -1, 7}, 
    {{1, 64, 64, 4}, -64, 320, 5}, 
    {{1, 64, 64, 8}, -64, 384, 11}, 
    {{1, 3, 3, 8}, 220064, -1, 5}, 
    {{1, 8, 1, 1}, 220144, -1, -1}, 
    {{1, 32, 32, 8}, -32, 448, 12}, 
    {{1, 32, 32, 12}, -32, 480, 13}, 
    {{4, 2, 1, 1}, 220176, -1, 7}, 
    {{1, 32, 32, 16}, -32, 512, 14}, 
    {{1, 32, 32, 14}, -32, 544, 15}, 
    {{1, 3, 3, 14}, 220208, -1, 5}, 
    {{1, 14, 1, 1}, 220336, -1, -1}, 
    {{1, 32, 32, 14}, -32, 576, 16}, 
    {{1, 32, 32, 16}, -32, 608, 17}, 
    {{1, 32, 32, 16}, -32, 640, 18}, 
    {{1, 32, 32, 28}, -32, 672, 19}, 
    {{1, 3, 3, 28}, 220400, -1, 5}, 
    {{1, 28, 1, 1}, 220656, -1, -1}, 
    {{1, 16, 16, 28}, -16, 704, 20}, 
    {{1, 16, 16, 32}, -16, 720, 21}, 
    {{1, 16, 16, 96}, -16, 736, 22}, 
    {{1, 3, 3, 96}, 220768, -1, 5}, 
    {{1, 96, 1, 1}, 221632, -1, -1}, 
    {{1, 16, 16, 96}, -16, 752, 23}, 
    {{1, 16, 16, 28}, -16, 768, 24}, 
    {{1, 16, 16, 32}, -16, 784, 25}, 
    {{1, 16, 16, 32}, -16, 800, 26}, 
    {{1, 16, 16, 96}, -16, 816, 27}, 
    {{1, 3, 3, 96}, 222016, -1, 5}, 
    {{1, 96, 1, 1}, 222880, -1, -1}, 
    {{1, 16, 16, 96}, -16, 832, 28}, 
    {{1, 16, 16, 28}, -16, 848, 29}, 
    {{1, 16, 16, 32}, -16, 864, 30}, 
    {{1, 16, 16, 32}, -16, 880, 31}, 
    {{1, 16, 16, 152}, -16, 896, 32}, 
    {{1, 3, 3, 152}, 223264, -1, 5}, 
    {{1, 152, 1, 1}, 224640, -1, -1}, 
    {{1, 8, 8, 152}, -8, 912, 33}, 
    {{1, 8, 8, 44}, -8, 920, 34}, 
    {{1, 8, 8, 268}, -8, 928, 35}, 
    {{1, 3, 3, 268}, 225248, -1, 5}, 
    {{1, 268, 1, 1}, 227664, -1, -1}, 
    {{1, 8, 8, 268}, -8, 936, 36}, 
    {{1, 8, 8, 40}, -8, 944, 37}, 
    {{1, 8, 8, 44}, -8, 952, 38}, 
    {{1, 8, 8, 44}, -8, 960, 39}, 
    {{1, 8, 8, 268}, -8, 968, 40}, 
    {{4, 2, 1, 1}, 228736, -1, 7}, 
    {{1, 8, 8, 52}, -8, 976, 41}, 
    {{1, 3, 3, 268}, 228768, -1, 5}, 
    {{1, 268, 1, 1}, 231184, -1, -1}, 
    {{1, 8, 8, 268}, -8, 984, 42}, 
    {{1, 8, 8, 52}, -8, 992, 43}, 
    {{1, 8, 8, 52}, -8, 1000, 44}, 
    {{1, 8, 8, 268}, -8, 1008, 45}, 
    {{1, 3, 3, 268}, 232256, -1, 5}, 
    {{1, 268, 1, 1}, 234672, -1, -1}, 
    {{1, 8, 8, 268}, -8, 1016, 46}, 
    {{1, 8, 8, 44}, -8, 1024, 47}, 
    {{1, 8, 8, 52}, -8, 1032, 48}, 
    {{1, 8, 8, 52}, -8, 1040, 49}, 
    {{1, 8, 8, 192}, -8, 1048, 50}, 
    {{1, 3, 3, 192}, 235744, -1, 5}, 
    {{1, 192, 1, 1}, 237472, -1, -1}, 
    {{1, 8, 8, 192}, -8, 1056, 51}, 
    {{1, 8, 8, 68}, -8, 1064, 5}, 
    {{1, 8, 8, 172}, -8, 1072, 52}, 
    {{1, 3, 3, 172}, 238240, -1, 5}, 
    {{1, 172, 1, 1}, 239792, -1, -1}, 
    {{1, 8, 8, 172}, -8, 1080, 53}, 
    {{1, 8, 8, 68}, -8, 1088, 54}, 
    {{1, 8, 8, 68}, -8, 1096, 55}, 
    {{1, 8, 8, 172}, -8, 1104, 56}, 
    {{1, 3, 3, 172}, 240480, -1, 5}, 
    {{1, 172, 1, 1}, 242032, -1, -1}, 
    {{1, 8, 8, 172}, -8, 1112, 57}, 
    {{1, 8, 8, 48}, -8, 1120, 58}, 
    {{4, 2, 1, 1}, 242720, -1, 7}, 
    {{1, 8, 8, 68}, -8, 1128, 59}, 
    {{1, 8, 8, 68}, -8, 1136, 60}, 
    {{1, 8, 8, 288}, -8, 1144, 61}, 
    {{1, 3, 3, 288}, 242752, -1, 5}, 
    {{1, 288, 1, 1}, 245344, -1, -1}, 
    {{1, 4, 4, 288}, -4, 1152, 62}, 
    {{1, 4, 4, 16}, -4, 1156, 63}, 
    {{1, 4, 4, 96}, -4, 1160, 64}, 
    {{1, 3, 3, 96}, 246496, -1, 5}, 
    {{1, 96, 1, 1}, 247360, -1, -1}, 
    {{1, 4, 4, 96}, -4, 1164, 65}, 
    {{1, 4, 4, 16}, -4, 1168, 66}, 
    {{1, 4, 4, 16}, -4, 1172, 67}, 
    {{4, 2, 1, 1}, 247744, -1, 7}, 
    {{1, 4, 4, 32}, -4, 1176, 68}, 
    {{1, 4, 4, 96}, -4, 1180, 69}, 
    {{1, 3, 3, 96}, 247776, -1, 5}, 
    {{1, 96, 1, 1}, 248640, -1, -1}, 
    {{1, 4, 4, 96}, -4, 1184, 70}, 
    {{1, 4, 4, 32}, -4, 1188, 5}, 
    {{1, 4, 4, 32}, -4, 1192, 71}, 
    {{1, 4, 4, 96}, -4, 1196, 72}, 
    {{1, 3, 3, 96}, 249024, -1, 5}, 
    {{1, 96, 1, 1}, 249888, -1, -1}, 
    {{1, 4, 4, 96}, -4, 1200, 73}, 
    {{1, 4, 4, 32}, -4, 1204, 74}, 
    {{1, 4, 4, 128}, -4, 1208, 75}, 
    {{1, 4, 4, 128}, -1, 0, 1}, 
    {{1, 1, 1, 128}, -1, 2048, 2}, 
    {{1, 1, 1, 2}, -1, 0, 3}, 
    {{1, 2, 1, 1}, 250272, -1, 7}, 
    {{1, 2, 1, 1}, -1, 16, 4}
};
const int all_0_zp_cursor = 5;
const int only_zp_cursor = 7;
const int only_zp_start = 293;
const int32_t quant_scale[7] = {
    1006982283, 1017742230, 1017742230, 1018604810, 1018604810, 974668598, 978780452
};
const int8_t quant_zeropoint[362] = {
    -70, -128, -128, -17, -17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -70, -128, -128, -128, -128, -128, -30, -30, -128, -128, -33, -13, -128, -128, -4, -128, -128, 2, 2, 1, -128, -128, 2, 2, 5, -128, -128, -13, -128, -128, 3, 3, -1, -128, -1, -128, -6, -13, -128, -128, -6, -6, 7, -128, -128, -128, -128, -12, -10, -128, -128, 18, 18, -8, -128, -128, -21, -128, -128, -10, -23, -23, -128, -128, 20, -128, -128, 1, -128
};
const int32_t split_offset[1212] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936, 8192, 8448, 8704, 8960, 9216, 9472, 9728, 9984, 10240, 10496, 10752, 11008, 11264, 11520, 11776, 12032, 12288, 12544, 12800, 13056, 13312, 13568, 13824, 14080, 14336, 14592, 14848, 15104, 15360, 15616, 15872, 16128, 16384, 16640, 0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936, 8192, 8448, 8704, 8960, 9216, 9472, 9728, 9984, 10240, 10496, 10752, 11008, 11264, 11520, 11776, 12032, 12288, 12544, 12800, 13056, 13312, 13568, 13824, 14080, 14336, 14592, 14848, 15104, 15360, 15616, 15872, 16128, 16384, 16640, 16896, 17152, 17408, 17664, 17920, 18176, 18432, 18688, 18944, 19200, 19456, 19712, 19968, 20224, 20480, 20736, 20992, 21248, 21504, 21760, 22016, 22272, 22528, 22784, 23040, 23296, 23552, 23808, 24064, 24320, 24576, 24832, 25088, 25344, 25600, 25856, 26112, 26368, 26624, 26880, 27136, 27392, 27648, 27904, 28160, 28416, 28672, 28928, 29184, 29440, 29696, 29952, 30208, 30464, 30720, 31232, 31744, 32256, 32768, 0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 0, 256, 2432, 2944, 3456, 3968, 4480, 4992, 5504, 6016, 6528, 7040, 7552, 8064, 8576, 9088, 9600, 10112, 10624, 11136, 11648, 12160, 12672, 13184, 13696, 14208, 14720, 15232, 15744, 16256, 16512, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 30720, 0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16832, 17280, 17728, 18176, 18624, 19072, 19520, 19968, 20416, 20864, 21312, 21760, 22208, 22656, 23104, 23552, 24000, 24448, 24896, 25344, 25792, 26240, 26688, 27136, 27584, 28032, 28480, 28928, 29376, 29824, 30272, 30720, 31168, 31616, 32064, 32512, 32960, 33408, 33856, 34304, 34752, 35200, 35648, 36096, 36544, 36992, 37440, 37888, 38336, 38784, 39232, 39680, 40128, 40576, 41024, 41472, 41920, 42368, 42816, 43264, 43712, 44160, 44608, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 0, 896, 1792, 2688, 3584, 4480, 5376, 6272, 7168, 8064, 8960, 9856, 10752, 11648, 12544, 13440, 14336, 15232, 16128, 17024, 17920, 18816, 19712, 20608, 21504, 22400, 23296, 24192, 25088, 25984, 26880, 27776, 28672, 0, 448, 896, 1344, 1792, 2240, 2688, 3136, 3584, 4032, 4480, 4928, 5376, 5824, 6272, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 33280, 33792, 34304, 34816, 35328, 0, 1536, 3072, 4608, 6144, 7680, 9216, 10752, 12288, 13824, 15360, 16896, 18432, 19968, 21504, 23040, 24576, 26112, 0, 1536, 3072, 4608, 6144, 7680, 9216, 10752, 12288, 13824, 15360, 16896, 18432, 19968, 21504, 21952, 22400, 22848, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 35840, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 33280, 33792, 34304, 34816, 0, 1536, 3072, 4608, 6144, 7680, 9216, 10752, 12288, 13824, 15360, 16896, 18432, 19968, 21504, 23040, 24576, 26112, 0, 1536, 3072, 4608, 6144, 7680, 9216, 10752, 12288, 13824, 15360, 16896, 18432, 19968, 21504, 21952, 22400, 22848, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 12160, 14592, 17024, 19456, 21888, 24320, 26752, 29184, 31616, 35328, 36480, 38912, 0, 2432, 4864, 7296, 9728, 12160, 14592, 17024, 19456, 21888, 24320, 26752, 29184, 31616, 34048, 36480, 38912, 0, 1216, 2432, 3648, 4864, 6080, 7296, 21440, 21792, 22144, 22496, 22848, 23200, 23552, 23904, 0, 2144, 4288, 6432, 8576, 10720, 12864, 15008, 17152, 19296, 0, 2144, 4288, 6432, 8576, 10720, 12864, 13184, 13504, 1408, 1760, 2112, 2464, 2816, 0, 352, 704, 1056, 1408, 1760, 2112, 2464, 2816, 4288, 6432, 8576, 10720, 12864, 15008, 17152, 0, 2144, 4288, 6432, 8576, 10720, 12864, 15008, 21440, 21856, 22272, 22688, 23104, 23520, 23936, 24352, 17152, 19296, 0, 2144, 4288, 6432, 8576, 10720, 12864, 13280, 13696, 0, 416, 832, 1248, 1664, 24768, 21440, 21856, 22272, 22688, 23104, 23520, 23936, 0, 2144, 4288, 6432, 8576, 10720, 12864, 15008, 17152, 19296, 0, 2144, 4288, 6432, 8576, 10720, 12864, 13216, 13568, 1664, 2080, 2496, 2912, 3328, 0, 416, 832, 1248, 1664, 2080, 2496, 2912, 3328, 3744, 4608, 6144, 7680, 9216, 10752, 12288, 0, 1536, 3072, 4608, 6144, 7680, 9216, 10752, 12288, 13824, 0, 1536, 3072, 4608, 6144, 7680, 15360, 15904, 13760, 14304, 16448, 16992, 17536, 18080, 0, 1376, 2752, 4128, 5504, 6880, 8256, 9632, 11008, 12384, 0, 1376, 2752, 4128, 5504, 6880, 8256, 8800, 9344, 0, 544, 1088, 1632, 2176, 18624, 14848, 15392, 13760, 14304, 15936, 16480, 17024, 0, 1376, 2752, 4128, 5504, 6880, 8256, 9632, 11008, 12384, 0, 1376, 2752, 4128, 5504, 6880, 8256, 8640, 9024, 9408, 9792, 3264, 3808, 4352, 0, 544, 1088, 1632, 2176, 2720, 3264, 3808, 4352, 4896, 6912, 9216, 11520, 13824, 17568, 18432, 0, 2304, 4608, 6912, 9216, 11520, 13824, 16128, 18432, 0, 1152, 2304, 3456, 3520, 3584, 3648, 0, 384, 768, 1152, 1536, 1920, 0, 384, 768, 832, 896, 0, 384, 768, 1152, 1536, 2304, 2432, 2560, 2688, 0, 384, 768, 1152, 1536, 1920, 0, 384, 768, 896, 1024, 0, 384, 768, 1152, 1536, 0, 384, 768, 1152, 1536, 1920, 0, 384, 768, 896, 1024, 0, 2048, 2560, 3072, 3584
};
alignas(16) const uint8_t offline_tensor_data[250288] = {
};

int CtxSummary(){
    printf("Arena Size: %d\n", arena_size);
    printf("Tensor Metadata Summary:\n");
    printf("\ttensors: %d\n",sizeof(tensors));
    printf("\tquant_scale: %d\n",21692);
    printf("\tquant_zeropoint: %d\n",362);
    printf("\tsplit_offset: %d\n",sizeof(split_offset));
    printf("\toffline_tensor_data: %d\n",sizeof(offline_tensor_data));

    int byte_tensor =   sizeof(tensors) +
                        21692 + 362 +
                        sizeof(split_offset) + sizeof(offline_tensor_data);
    return byte_tensor;
}