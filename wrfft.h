#ifndef WRFFT_H
#define WRFFT_H

#include <stddef.h>   // for size_t
#include "wrfft_common.h"
#include "cufft_wrapper/cufft_common.h"
#include "cufft_wrapper/cufft_wrapper_1d.h"
#include "cufft_wrapper/cufft_wrapper_2d.h"

// Library version
#define WRFFT_VERSION "1.0.0"

// Optimization criteria that can be used in planning
typedef enum {
    OPTIMIZE_SPEED,          // Optimize for speed
    OPTIMIZE_ENERGY          // Optimize for energy efficiency
} WrFFTOptimizationCriteria;

// Basic complex data type for the API.
typedef struct {
    double real;
    double imag;
} ComplexData;


// Main configuration structure for FFT operations.
// This structure is partially set by the user and partially populated internally by the API.
typedef struct {
    size_t data_size[3];                   // Data size: [Nx, Ny, Num_dimensions]
    WrFFTOptimizationCriteria criteria;    // Optimization criterion selected for the transform
    double error_threshold;                // User-defined maximum allowed prediction error
    const char *chosen_library;            // Internally set: Name of the selected FFT library
    const char *chosen_precision;          // Internally set: Selected precision ("half", "single", "double")
    void *internal_plan;                   // Internally managed plan pointer (opaque to the user)
    void *gpu_input;                       // Internally managed pointer to GPU input data
    void *gpu_output;                      // Internally managed pointer to GPU output data
} WrFFTConfig;

/**
 * @brief Initializes the wrFFT library, loading all models and FFT back-ends,
 *        and sets a custom prediction error threshold and optimization goal.
 * This:
 *  1) loads all models/back-ends just once,
 *  2) populates cfg->data_size = {nx, ny, nd}, 
 *  3) sets cfg->error_threshold and cfg->criteria,
 *  4) leaves cfg ready for classify/plan/execute/finalize.
 *
 * @param nx               FFT length in X (e.g. N for 1-D)
 * @param ny               FFT length in Y (use 1 for 1-D)
 * @param error_threshold  Maximum tolerated ML-prediction error.
 * @param criteria         Which metric to optimize (accuracy/speed/energy).
 * @param[out] cfg         Uninitialized config; will be filled in.
 * @return WrFFTErrors     WRFFT_SUCCESS on success.
 */
WrFFTErrors wrfft_initialize(size_t nx, size_t ny, double error_threshold, WrFFTOptimizationCriteria goal, WrFFTConfig *cfg);

/**
 * @brief Classifies the input FFT data and determines the optimal FFT library and precision.
 * 
 * This function extracts features from the input data (passed as an array of ComplexData),
 * then runs the classification model (e.g., via ONNX) to choose the best library/precision.
 * The decision is stored in the provided configuration.
 *
 * @param input_data Pointer to the input complex data.
 * @param config     Pointer to the WrFFT configuration structure, which will be updated with the chosen library and precision.
 * @return WrFFTErrors WRFFT_SUCCESS if classification succeeds, or an appropriate error code otherwise.
 */
WrFFTErrors wrfft_classify(ComplexData *input_data, WrFFTConfig *config);

/**
 * @brief Creates and configures an FFT plan using the optimal FFT library.
 * 
 * This function allocates necessary resources and prepares a plan for execution.
 * The plan is stored in the WrFFT configuration structure.
 *
 * @param input_data Pointer to the input complex data.
 * @param config     Pointer to the WrFFT configuration. Must have been updated by wrfft_classify().
 * @return WrFFTErrors WRFFT_SUCCESS if plan creation succeeds, or an appropriate error code otherwise.
 */
WrFFTErrors wrfft_plan(ComplexData *input_data, WrFFTConfig *config);

/**
 * @brief Executes the FFT using the selected and planned FFT library.
 * 
 * The output is stored in the provided output data array.
 *
 * @param output_data Pointer to an array where the FFT output will be stored.
 * @param config      Pointer to the WrFFT configuration.
 * @return WrFFTErrors WRFFT_SUCCESS if the FFT execution succeeds, or an appropriate error code otherwise.
 */
WrFFTErrors wrfft_execute(ComplexData *output_data, WrFFTConfig *config);

/**
 * @brief Finalizes the FFT operations and cleans up all allocated resources.
 * 
 * This function should be called at the end of the program to free internal memory and resources.
 *
 * @param config Pointer to the WrFFT configuration used for planning/execution.
 * @return WrFFTErrors WRFFT_SUCCESS if cleanup succeeds, or an appropriate error code otherwise.
 */
WrFFTErrors wrfft_finalize(WrFFTConfig *config);

#endif // WRFFT_H
