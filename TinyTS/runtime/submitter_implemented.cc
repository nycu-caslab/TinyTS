#include "api/submitter_implemented.h"
#include "api/internally_implemented.h"
#include "util/quantization_helpers.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mbed.h"

#include "gen_lib/include/ctx_util.h"
#include "gen_model/include/ctx.h"
#include "gen_model/include/eval.h"

// my vars
#include <climits>
int32_t tot_time = 0;
int32_t latency_max = INT32_MIN;
int32_t latency_min = INT32_MAX;

UnbufferedSerial pc(USBTX, USBRX);
DigitalOut timestampPin(D7);

#ifdef EVICT_IN
int8_t input_arr[224*224*3] = {0};
#endif

// Implement this method to prepare for inference and preprocess inputs.
void th_load_tensor() {
  // int input_size = GetTensorSize(input_tid);
  // int8_t input_asint[input_size];
  
  // input from MLPerf interface
  // uint8_t input_quantized[input_size];
  // size_t bytes = ee_get_buffer(reinterpret_cast<uint8_t *>(input_quantized),
  //                              input_size * sizeof(uint8_t));
  // if (bytes / sizeof(uint8_t) != input_size) {
  //   th_printf("Input db has %d elemented, expected %d\n", bytes / sizeof(uint8_t),
  //             input_size);
  //   return;
  // }

  // input from fixed input
  // uint8_t *input_quantized = const_cast<uint8_t*>(rnd_input);

  // uint16_t i = 0;
  // for(i=0; i<input_size;i++)
  // {
	//   if(input_quantized[i]<=127)
	//     input_asint[i] = ((int8_t)input_quantized[i]) - 128;
	//   else
	//     input_asint[i] = (int8_t)(input_quantized[i] - 128);
  // }

  // fill input tensor with zero
  // int i = 0;
  // for(i=0; i<input_size;i++)
  // {
	//   input_asint[i] = 0;
  // }

  // fill_input_tensor(input_asint);
#ifdef EVICT_IN
  for(int i = 0; i< 224*224*3; i++) 
    input_arr[i] = 0;
#else
  fill_input_tensor_w_val(0);
#endif
}

// Add to this method to return real inference results.
void th_results() {
  // const int nresults = 10;
  /**
   * The results need to be printed back in exactly this format; if easier
   * to just modify this loop than copy to results[] above, do that.
   */
  th_printf("m-results-[");
#if HIDE_OUTPUT
  th_printf("Result is hidden by macro.");
#else
  int kCategoryCount = GetTensorSize(output_tid);
  int8_t *output_data = GetTensorData(output_tid);
  const float  output_scale = GetTensorQuantScale(output_tid)[0];
  const int32_t  output_zero_point = GetTensorQuantZP(output_tid)[0];

  for (size_t i = 0; i < kCategoryCount; i++) {
    // print dequantized result
    // float converted =
    //     DequantizeInt8ToFloat(output_data[i], output_scale,
    //                           output_zero_point);
    // th_printf("%0.3f", converted);

    // print quantized result
    int converted = output_data[i];
    th_printf("%d", converted);
    if (i < (kCategoryCount - 1)) {
      th_printf(",");
    }
  }
#endif
  th_printf("]\r\n");
}
// Implement this method with the logic to perform one inference cycle.
void th_infer() { 
  eval(input_arr);
}

/// \brief optional API.
void th_final_initialize(void) {
  CodeGenSummary();
}
void th_pre() {}
void th_post() {}

void th_command_ready(char volatile *p_command) {
  p_command = p_command;
  ee_serial_command_parser_callback((char *)p_command);
}

// th_libc implementations.
int th_strncmp(const char *str1, const char *str2, size_t n) {
  return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n) {
  return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen) {
  return strnlen(str, maxlen);
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }

char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }

int th_atoi(const char *str) { return atoi(str); }

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}

/* N.B.: Many embedded *printf SDKs do not support all format specifiers. */
int th_vprintf(const char *format, va_list ap) { return vprintf(format, ap); }
void th_printf(const char *p_fmt, ...) {
  va_list args;
  va_start(args, p_fmt);
  (void)th_vprintf(p_fmt, args); /* ignore return */
  va_end(args);
}

char th_getchar() { return getchar(); }

void th_serialport_initialize(void) {
#if EE_CFG_ENERGY_MODE==1
  pc.baud(9600);
#else
  pc.baud(115200);
#endif
}

uint32_t th_timestamp(void) {
  # if EE_CFG_ENERGY_MODE==1
  timestampPin = 0;
  for (int i=0; i<100'000; ++i) {
    asm("nop");
  }
  timestampPin = 1;
 #else
  unsigned long microSeconds = 0ul;
  /* USER CODE 2 BEGIN */
  microSeconds = us_ticker_read();
  /* USER CODE 2 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP, microSeconds);
  #endif
  return microSeconds;
}

void th_timestamp_initialize(void) {
  /* USER CODE 1 BEGIN */
  // Setting up BOTH perf and energy here
  /* USER CODE 1 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP_MODE);
  /* Always call the timestamp on initialize so that the open-drain output
     is set to "1" (so that we catch a falling edge) */
  th_timestamp();
}
