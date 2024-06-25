#ifndef GEN_INCLUDE_BUILTIN_OP_DATA_H_
#define GEN_INCLUDE_BUILTIN_OP_DATA_H_

typedef enum {
  /* kTfLitePaddingUnknown = 0, */
  kTfLitePaddingSame = 0,
  kTfLitePaddingValid,
} TfLitePadding;

/* TODO */
typedef struct {
  int width;
  int height;
  int width_offset;
  int height_offset;
} PaddingValues;

// Possible fused activation functions.
// TODO(aselle): rename to TfLiteActivation
typedef enum {
  kTfLiteActNone = 0,
  kTfLiteActRelu,
  kTfLiteActReluN1To1,                    // min(max(-1, x), 1)
  kTfLiteActRelu1 = kTfLiteActReluN1To1,  // kTfLiteActRelu1 will be deprecated.
  kTfLiteActRelu6,                        // min(max(0, x), 6)
  kTfLiteActTanh,
  kTfLiteActSignBit,
  kTfLiteActSigmoid,
} TfLiteFusedActivation;

#endif