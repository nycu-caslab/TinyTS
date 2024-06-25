/*
 * mbed SDK
 * Copyright (c) 2017 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Automatically generated configuration file.
// DO NOT EDIT, content will be overwritten.

#ifndef __MBED_CONFIG_DATA__
#define __MBED_CONFIG_DATA__

// Configuration parameters
#define CLOCK_SOURCE                                                      USE_PLL_MSI            // set by target:MCU_STM32L4
#define LPTICKER_DELAY_TICKS                                              0                      // set by target:MCU_STM32L4
#define MBED_CONF_APP_MAIN_STACK_SIZE                                     20480                  // set by application
#define MBED_CONF_APP_THREAD_STACK_SIZE                                   20480                  // set by application
#define MBED_CONF_DRIVERS_OSPI_CSN                                        OSPI_FLASH1_CSN        // set by library:drivers
#define MBED_CONF_DRIVERS_OSPI_DQS                                        OSPI_FLASH1_DQS        // set by library:drivers
#define MBED_CONF_DRIVERS_OSPI_IO0                                        OSPI_FLASH1_IO0        // set by library:drivers
#define MBED_CONF_DRIVERS_OSPI_IO1                                        OSPI_FLASH1_IO1        // set by library:drivers
#define MBED_CONF_DRIVERS_OSPI_IO2                                        OSPI_FLASH1_IO2        // set by library:drivers
#define MBED_CONF_DRIVERS_OSPI_IO3                                        OSPI_FLASH1_IO3        // set by library:drivers
#define MBED_CONF_DRIVERS_OSPI_IO4                                        OSPI_FLASH1_IO4        // set by library:drivers
#define MBED_CONF_DRIVERS_OSPI_IO5                                        OSPI_FLASH1_IO5        // set by library:drivers
#define MBED_CONF_DRIVERS_OSPI_IO6                                        OSPI_FLASH1_IO6        // set by library:drivers
#define MBED_CONF_DRIVERS_OSPI_IO7                                        OSPI_FLASH1_IO7        // set by library:drivers
#define MBED_CONF_DRIVERS_OSPI_SCK                                        OSPI_FLASH1_SCK        // set by library:drivers
#define MBED_CONF_DRIVERS_QSPI_CSN                                        QSPI_FLASH1_CSN        // set by library:drivers
#define MBED_CONF_DRIVERS_QSPI_IO0                                        QSPI_FLASH1_IO0        // set by library:drivers
#define MBED_CONF_DRIVERS_QSPI_IO1                                        QSPI_FLASH1_IO1        // set by library:drivers
#define MBED_CONF_DRIVERS_QSPI_IO2                                        QSPI_FLASH1_IO2        // set by library:drivers
#define MBED_CONF_DRIVERS_QSPI_IO3                                        QSPI_FLASH1_IO3        // set by library:drivers
#define MBED_CONF_DRIVERS_QSPI_SCK                                        QSPI_FLASH1_SCK        // set by library:drivers
#define MBED_CONF_DRIVERS_UART_SERIAL_RXBUF_SIZE                          256                    // set by library:drivers
#define MBED_CONF_DRIVERS_UART_SERIAL_TXBUF_SIZE                          256                    // set by library:drivers
#define MBED_CONF_PLATFORM_CALLBACK_COMPARABLE                            1                      // set by library:platform
#define MBED_CONF_PLATFORM_CALLBACK_NONTRIVIAL                            0                      // set by library:platform
#define MBED_CONF_PLATFORM_CRASH_CAPTURE_ENABLED                          0                      // set by library:platform
#define MBED_CONF_PLATFORM_CTHUNK_COUNT_MAX                               8                      // set by library:platform
#define MBED_CONF_PLATFORM_DEEPSLEEP_STATS_VERBOSE                        0                      // set by library:platform[STM]
#define MBED_CONF_PLATFORM_DEFAULT_SERIAL_BAUD_RATE                       9600                   // set by library:platform
#define MBED_CONF_PLATFORM_ERROR_ALL_THREADS_INFO                         0                      // set by library:platform
#define MBED_CONF_PLATFORM_ERROR_FILENAME_CAPTURE_ENABLED                 0                      // set by library:platform
#define MBED_CONF_PLATFORM_ERROR_HIST_ENABLED                             0                      // set by library:platform
#define MBED_CONF_PLATFORM_ERROR_HIST_SIZE                                4                      // set by library:platform
#define MBED_CONF_PLATFORM_ERROR_REBOOT_MAX                               1                      // set by library:platform
#define MBED_CONF_PLATFORM_FATAL_ERROR_AUTO_REBOOT_ENABLED                0                      // set by library:platform
#define MBED_CONF_PLATFORM_MAX_ERROR_FILENAME_LEN                         16                     // set by library:platform
#define MBED_CONF_PLATFORM_MINIMAL_PRINTF_ENABLE_64_BIT                   0                      // set by application[*]
#define MBED_CONF_PLATFORM_MINIMAL_PRINTF_ENABLE_FLOATING_POINT           1                      // set by application[*]
#define MBED_CONF_PLATFORM_MINIMAL_PRINTF_SET_FLOATING_POINT_MAX_DECIMALS 6                      // set by application[*]
#define MBED_CONF_PLATFORM_POLL_USE_LOWPOWER_TIMER                        0                      // set by library:platform
#define MBED_CONF_PLATFORM_STDIO_BAUD_RATE                                9600                   // set by library:platform
#define MBED_CONF_PLATFORM_STDIO_BUFFERED_SERIAL                          0                      // set by library:platform
#define MBED_CONF_PLATFORM_STDIO_CONVERT_NEWLINES                         1                      // set by library:platform
#define MBED_CONF_PLATFORM_STDIO_CONVERT_TTY_NEWLINES                     1                      // set by library:platform
#define MBED_CONF_PLATFORM_STDIO_FLUSH_AT_EXIT                            1                      // set by library:platform
#define MBED_CONF_PLATFORM_STDIO_MINIMAL_CONSOLE_ONLY                     0                      // set by library:platform
#define MBED_CONF_PLATFORM_USE_MPU                                        1                      // set by library:platform
#define MBED_CONF_RTOS_API_PRESENT                                        1                      // set by library:rtos-api
#define MBED_CONF_TARGET_BOOT_STACK_SIZE                                  0x1000                 // set by target:Target
#define MBED_CONF_TARGET_CONSOLE_UART                                     1                      // set by target:Target
#define MBED_CONF_TARGET_CUSTOM_TICKERS                                   1                      // set by target:Target
#define MBED_CONF_TARGET_DEEP_SLEEP_LATENCY                               4                      // set by target:MCU_STM32
#define MBED_CONF_TARGET_DEFAULT_ADC_VREF                                 NAN                    // set by target:Target
#define MBED_CONF_TARGET_GPIO_RESET_AT_INIT                               0                      // set by target:MCU_STM32
#define MBED_CONF_TARGET_I2C_TIMING_VALUE_ALGO                            0                      // set by target:MCU_STM32L4
#define MBED_CONF_TARGET_INIT_US_TICKER_AT_BOOT                           1                      // set by target:MCU_STM32
#define MBED_CONF_TARGET_INTERNAL_FLASH_UNIFORM_SECTORS                   1                      // set by target:Target
#define MBED_CONF_TARGET_LPTICKER_LPTIM                                   1                      // set by target:MCU_STM32L4
#define MBED_CONF_TARGET_LPTICKER_LPTIM_CLOCK                             1                      // set by target:MCU_STM32
#define MBED_CONF_TARGET_LPUART_CLOCK_SOURCE                              USE_LPUART_CLK_HSI     // set by target:MCU_STM32L496xG
#define MBED_CONF_TARGET_LSE_AVAILABLE                                    1                      // set by target:MCU_STM32
#define MBED_CONF_TARGET_LSE_DRIVE_LOAD_LEVEL                             RCC_LSEDRIVE_LOW       // set by target:MCU_STM32L4
#define MBED_CONF_TARGET_MPU_ROM_END                                      0x0fffffff             // set by target:Target
#define MBED_CONF_TARGET_RTC_CLOCK_SOURCE                                 USE_RTC_CLK_LSE_OR_LSI // set by target:MCU_STM32
#define MBED_CONF_TARGET_TICKLESS_FROM_US_TICKER                          0                      // set by target:Target
#define MBED_CONF_TARGET_XIP_ENABLE                                       0                      // set by target:Target
#define MBED_CRC_TABLE_SIZE                                               16                     // set by library:drivers
#define MBED_STACK_DUMP_ENABLED                                           0                      // set by library:platform
#define MBED_TRACE_COLOR_THEME                                            0                      // set by library:mbed-trace
#define MEM_ALLOC                                                         malloc                 // set by library:mbed-trace
#define MEM_FREE                                                          free                   // set by library:mbed-trace
// Macros
#define CMSIS_NN                                                                                 // defined by application
#define TF_LITE_STATIC_MEMORY                                                                    // defined by application
#define TF_LITE_USE_CTIME                                                                        // defined by application

#endif
