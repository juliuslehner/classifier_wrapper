#ifndef CUFFTDX_COMMON_HPP
#define CUFFTDX_COMMON_HPP

/// run-time precision selector for the cufftdx wrapper:
typedef enum {
    CUFFTDX_PREC_SINGLE = 0,
    CUFFTDX_PREC_DOUBLE = 1,
    CUFFTDX_PREC_HALF   = 2
} CufftdxPrecision;

#endif // CUFFTDX_COMMON_HPP
