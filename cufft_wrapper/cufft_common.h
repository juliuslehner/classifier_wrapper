#ifndef CUFFT_COMMON_H
#define CUFFT_COMMON_H

/// Precision enums used by the cuFFT wrappers:
typedef enum CufftPrecision {
    CUFFT_PREC_SINGLE = 0,
    CUFFT_PREC_DOUBLE = 1,
    CUFFT_PREC_HALF   = 2
} CufftPrecision;

#endif // CUFFT_COMMON_H
