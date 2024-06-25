#ifndef GEN_INCLUDE_COMMON_H_
#define GEN_INCLUDE_COMMON_H_
#include <stdint.h>

// #ifdef __cplusplus
// extern "C" {
// #endif  // __cplusplus

#define DEBUG_HALT while(1){}
#define DEBUG_ERR_MSG(err_msg) printf("ERROR: %s() in %s:%d: %s", __FUNCTION__, __FILE__, __LINE__, err_msg);DEBUG_HALT
#define OPT_INLINE 0
#define OPT_16ARG 0
#define OPT_OFFLOAD_ENABLE 1
#define OPT_INDEX 1
#define HIDE_OUTPUT 0
typedef enum TfLiteStatus {
  kTfLiteOk = 0,
  kTfLiteError = 1,
  kTfLiteDelegateError = 2
} TfLiteStatus;

#define TF_LITE_ENSURE_STATUS(a) \
  do {                           \
    const TfLiteStatus s = (a);  \
    if (s != kTfLiteOk) {        \
      return s;                  \
    }                            \
  } while (0)

// typedef enum TensorDataType: int8_t {
//     kFLOAT32=0, 
//     kINT32=1, 
//     kINT16=2, 
//     kINT8=3,
//     kUINT8=4
// }TensorDataType;

// SupportedQuantizationTypes.
typedef enum TfLiteQuantizationType {
  // No quantization.
  kTfLiteNoQuantization = 0,
  // Affine quantization (with support for per-channel quantization).
  // Corresponds to TfLiteAffineQuantization.
  kTfLiteAffineQuantization = 1,
} TfLiteQuantizationType;



// Fixed size list of floats. Used for per-channel quantization.
// typedef struct TfLiteFloatArray {
//   int size;
// // gcc 6.1+ have a bug where flexible members aren't properly handled
// // https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
// // This also applies to the toolchain used for Qualcomm Hexagon DSPs.
// #if !defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
//     __GNUC_MINOR__ >= 1
//   float data[0];
// #else
//   float data[];
// #endif
// } TfLiteFloatArray;

// // Fixed size list of integers. Used for dimensions and inputs/outputs tensor
// // indices
// typedef struct TfLiteIntArray {
//   int size;
// // gcc 6.1+ have a bug where flexible members aren't properly handled
// // https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
// #if (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
//      __GNUC_MINOR__ >= 1) ||                                      \
//     defined(HEXAGON)
//   int data[0];
// #else
//   int data[];
// #endif
// } TfLiteIntArray;

// Structure specifying the quantization used by the tensor, if-any.
typedef struct TfLiteQuantization {
  // The type of quantization held by params.
  TfLiteQuantizationType type;
  // Holds a reference to one of the quantization param structures specified
  // below.
  void* params;
} TfLiteQuantization;

// Legacy. Will be deprecated in favor of TfLiteAffineQuantization.
// If per-layer quantization is specified this field will still be populated in
// addition to TfLiteAffineQuantization.
// Parameters for asymmetric quantization. Quantized values can be converted
// back to float using:
//     real_value = scale * (quantized_value - zero_point)
typedef struct TfLiteQuantizationParams {
  float scale;
  int32_t zero_point;
} TfLiteQuantizationParams;

// Parameters for asymmetric quantization across a dimension (i.e per output
// channel quantization).
// quantized_dimension specifies which dimension the scales and zero_points
// correspond to.
// For a particular value in quantized_dimension, quantized values can be
// converted back to float using:
//     real_value = scale * (quantized_value - zero_point)
// typedef struct TfLiteAffineQuantization {
//   TfLiteFloatArray* scale;
//   TfLiteIntArray* zero_point;
//   int32_t quantized_dimension;
// } TfLiteAffineQuantization;

typedef struct Tensor {
    int dims[4];
    // TensorDataType type;
    int quant_offset;
    int data_offset;
    int split_offset;
    
    // Quantization information.
    // TfLiteQuantizationParams params;
    // Quantization information. Replaces params field above.
    // TfLiteQuantization quantization;
}Tensor;



// #endif  // __cplusplus
#endif