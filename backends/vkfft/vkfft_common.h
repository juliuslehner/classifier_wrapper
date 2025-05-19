#ifndef VKFFT_COMMON_H
#define VKFFT_COMMON_H

/// Precision enums used by the cuFFT wrappers:
typedef enum VkfftPrecision {
    VKFFT_PREC_SINGLE = 0,
    VKFFT_PREC_DOUBLE = 1,
    VKFFT_PREC_HALF   = 2
} VkfftPrecision;

#endif // VKFFT_COMMON_H
