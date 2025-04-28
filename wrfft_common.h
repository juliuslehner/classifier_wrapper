#ifndef WRFFT_COMMON_H
#define WRFFT_COMMON_H

/// Error codes used by the WrFFT API itself:
typedef enum WrFFTErrors {
    WRFFT_SUCCESS = 0,
    WRFFT_ERROR_INVALID_INPUT,
    WRFFT_ERROR_MODEL_FAILURE,
    WRFFT_ERROR_LIBRARY_FAILURE
} WrFFTErrors;

#endif // WRFFT_COMMON_H
