#ifndef PROF_CTRL_H_
#define PROF_CTRL_H_

// switches
#define PerOpCycleCount false

// includes
#if defined(STM32L496xx)
    #include <stm32l496xx.h>
#elif defined(STM32F767xx)
    #include <stm32f767xx.h>
#endif

// macro functions
#define DWT_CYCCNT_START() DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk
#define DWT_CYCCNT_STOP() DWT->CTRL &= ~DWT_CTRL_CYCCNTENA_Msk
#define DWT_CYCCNT_RESET() DWT->CYCCNT = 0
#define DWT_CYCCNT_GET() (DWT->CYCCNT)
#define DWT_CYCCNT_EN() CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk
#define DWT_CYCCNT_LOCK() DWT->LAR = 0
#define DWT_CYCCNT_UNLOCK() DWT->LAR = 0xC5ACCE55

// variables or literal macro defs
extern uint32_t cc_min;
extern uint32_t cc_max;
extern int64_t cc_tot;

// function decls

#endif