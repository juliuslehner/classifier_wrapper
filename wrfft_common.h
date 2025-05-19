#ifndef WRFFT_COMMON_H
#define WRFFT_COMMON_H

/// Error codes used by the WrFFT API itself:
typedef enum WrFFTErrors {
    WRFFT_SUCCESS = 0,
    WRFFT_ERROR_INVALID_INPUT,
    WRFFT_ERROR_MODEL_FAILURE,
    WRFFT_ERROR_LIBRARY_FAILURE
} WrFFTErrors;

// Precision enums used by the WrFFT wrappers:
typedef enum WrFFTPrecision {
    WRFFT_PREC_SINGLE = 0,
    WRFFT_PREC_DOUBLE = 1,
    WRFFT_PREC_HALF   = 2
} WrFFTPrecision;

// Basic complex data type for the API.
typedef struct {
    double x;   // Real part
    double y;   // Imaginary part
} ComplexData;

#endif // WRFFT_COMMON_H
