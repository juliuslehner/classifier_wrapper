#include "wrfft.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda_fp16.h>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <limits>

// --------------------------- Configurable Constants -------------------------
static const double EPSILON_SMALL      = 1e-16;  // avoid log underflow
static const size_t SAMPLE_SIZE        = 2000;   // feature extraction samples
static const char* DEFAULT_MODEL_DIR   = "model_files_onnx_log";
static const char* DEFAULT_PERF_JSON   = "performance_data_multidimensional.json";

// ------------------------ Internal Data Structures -------------------------
struct Features {
    float var_real;
    float var_imag;
    float avg_magnitude;
    size_t nx;
    size_t ny;
};

// ---------------------------------------------------------------------------
// Precision helper
// ---------------------------------------------------------------------------
static inline CufftPrecision to_enum(const char *txt)
{
    if (txt && std::strcmp(txt, "single") == 0)  return CUFFT_PREC_SINGLE;
    if (txt && std::strcmp(txt, "half")   == 0)  return CUFFT_PREC_HALF;
    return CUFFT_PREC_DOUBLE;                     // default / “double”
}

#ifndef CUDA_CHECK
#define CUDA_CHECK(err)                                               \
  do {                                                                 \
    cudaError_t e_ = (err);                                            \
    if (e_ != cudaSuccess) {                                           \
      std::cerr << "CUDA error " << cudaGetErrorString(e_)            \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";     \
      return WRFFT_ERROR_LIBRARY_FAILURE;                              \
    }                                                                  \
  } while(0)
#endif

// Simple key for models
struct ModelKey {
    std::string lib;
    std::string dim;
    std::string prec;
};

// Holds one ONNX model session and names
struct ModelInfo {
    ModelKey key;
    Ort::Session session;
    char* input_name;
    char* output_name;
};

// Global predictor state
static Ort::Env* ort_env = nullptr;
static Ort::SessionOptions* session_opts = nullptr;
static std::vector<ModelInfo> models;
static nlohmann::json perf_data;
static bool predictor_ready = false;

// --------------------------- Utility Functions ------------------------------

// Extract simple features: variances and average magnitude
Features extract_features(const ComplexData* data, size_t nx, size_t ny) {
    size_t n = nx * ny;
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937{12345});
    idx.resize(std::min(n, SAMPLE_SIZE));

    double sum_r=0, sum_i=0;
    for (auto i: idx) { sum_r += data[i].real; sum_i += data[i].imag; }
    double mean_r = sum_r/idx.size();
    double mean_i = sum_i/idx.size();

    double var_r=0, var_i=0, mag=0;
    for (auto i: idx) {
        double dr = data[i].real - mean_r;
        double di = data[i].imag - mean_i;
        var_r += dr*dr;
        var_i += di*di;
        mag   += std::hypot(data[i].real, data[i].imag);
    }
    var_r /= idx.size();
    var_i /= idx.size();
    mag   /= idx.size();

    return Features{(float)var_r, (float)var_i, (float)mag, nx, ny};
}

// Convert log10 error to actual error
static double to_error(float log10_err) {
    double val = pow(10.0, log10_err);
    return std::max(val, EPSILON_SMALL);
}

// Load performance JSON file
static void load_performance(const char* path) {
    std::ifstream f(path);
    perf_data = nlohmann::json::parse(f);
}

static double cost_function(const std::string& lib, const std::string& prec, size_t nx, size_t ny, WrFFTOptimizationCriteria goal) {
    // 1) build the top‐level key, e.g. "cufft_single"
    std::string mapkey = lib + "_" + prec;
    if (!perf_data.contains(mapkey)) {
        return std::numeric_limits<double>::infinity();
    }

    // 2) build the size‐tuple string exactly as your JSON does,
    //    e.g. "(1024,1)" or "(8,8)"
    std::ostringstream oss;
    oss << "(" << nx << "," << ny << ")";
    std::string size_key = oss.str();

    // 3) look up in the inner object
    auto& inner = perf_data[mapkey];
    if (!inner.contains(size_key)) {
        return std::numeric_limits<double>::infinity();
    }
    auto& metrics = inner[size_key];
    const double INF = std::numeric_limits<double>::infinity();

    if (goal == OPTIMIZE_SPEED) {
        return metrics.value("time", INF);
    } else {
        // look for the "energy" field
        return metrics.value("energy", INF);
    }
}

// Load ONNX models from directory
static void load_models(const char* dir) {
    ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "wrfft");
    Ort::SessionOptions so; 
    so.SetIntraOpNumThreads(1);
    try{ OrtSessionOptionsAppendExecutionProvider_CUDA(so,0);}catch(...){}

    for (auto& entry: std::filesystem::recursive_directory_iterator(dir)) {
        if (entry.path().extension() != ".onnx") continue;
        auto path = entry.path().string();
        ModelInfo info{ModelKey{}, Ort::Session(*ort_env, path.c_str(), so), nullptr, nullptr};
        // parse key from filename: lib_dim_prec.onnx
        auto stem = entry.path().stem().string();
        size_t p1 = stem.find('_'), p2 = stem.find('_', p1+1);
        info.key.lib  = stem.substr(0, p1);
        info.key.dim  = stem.substr(p1+1, p2-p1-1);
        info.key.prec = stem.substr(p2+1);
        // input/output names
        info.input_name  = info.session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).release();
        info.output_name = info.session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).release();
        models.push_back(std::move(info));
    }
}

// Initialize predictor: load JSON and models
WrFFTErrors init_predictor() {
        // 1) load the performance JSON
    try {
        load_performance(DEFAULT_PERF_JSON);
    } catch (const std::exception &e) {
        std::cerr << "[wrfft] ERROR loading JSON '" << DEFAULT_PERF_JSON
                  << "': " << e.what() << "\n";
        predictor_ready = false;
        return WRFFT_ERROR_MODEL_FAILURE;
    }

    // 2) load all the ONNX models
    try {
        load_models(DEFAULT_MODEL_DIR);
    } catch (const Ort::Exception &e) {
        std::cerr << "[wrfft] ONNX exception loading models from '"
                  << DEFAULT_MODEL_DIR << "': " << e.what() << "\n";
        predictor_ready = false;
        return WRFFT_ERROR_MODEL_FAILURE;
    }

    predictor_ready = true;
    return WRFFT_SUCCESS;
}

// Predict error given a model and feature vector
static double run_model(ModelInfo& m, const std::vector<float>& feat) {
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto dims = std::array<int64_t,2>{1, (int64_t)feat.size()};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mi, (float*)feat.data(), feat.size(), dims.data(), 2);
    const char* in = m.input_name;
    const char* out = m.output_name;
    auto result = m.session.Run(Ort::RunOptions{nullptr}, &in, &input_tensor, 1, &out, 1);
    float log10_err = *result.front().GetTensorMutableData<float>();
    return to_error(log10_err);
}

// Classify to choose best lib/precision
static void classify_model(const ComplexData* data, WrFFTConfig* cfg, char const*& lib, char const*& prec) {
    if (!predictor_ready) { lib="cufft"; prec="double"; return; }
    auto feat = extract_features(data, cfg->data_size[0], cfg->data_size[1]);
    std::vector<float> fv = {
        feat.var_real,
        feat.var_imag,
        feat.avg_magnitude,
        float(feat.nx)
    };
    if (feat.ny > 1) {
        fv.push_back(float(feat.ny));
    }
    std::string dim = (feat.ny>1 ? "2d" : "1d");
    struct Cand{double cost; const char* lib; const char* prec;};
    std::vector<Cand> cands;
    for (auto& m : models) {
        if (m.key.dim != dim) continue;
        double err = run_model(m, fv);
        if (err <= cfg->error_threshold) {
            std::string mapkey = m.key.lib + "_" + m.key.prec;
            double cost = cost_function(m.key.lib, m.key.prec, feat.nx, feat.ny, cfg->criteria);
            cands.push_back({cost, m.key.lib.c_str(), m.key.prec.c_str()});
        }
    }
    if (cands.empty()) { lib="cufft"; prec="double"; }
    else {
        auto best = *std::min_element(cands.begin(), cands.end(), [](auto&a,auto&b){return a.cost<b.cost;});
        lib = best.lib; prec = best.prec;
    }
}

// ----------------------------- WrFFT API ------------------------------------

WrFFTErrors wrfft_initialize(size_t nx, size_t ny, double threshold, WrFFTOptimizationCriteria goal, WrFFTConfig* cfg) {
    if (!cfg) return WRFFT_ERROR_INVALID_INPUT;

    // 1) one-time init of ML predictor/models
    WrFFTErrors e = init_predictor();
    if (e != WRFFT_SUCCESS) return e;

    // 2) fill in user config
    cfg->data_size[0]    = nx;
    cfg->data_size[1]    = ny;
    cfg->data_size[2]    = (ny > 1 ? 2 : 1);        // infer number of dimensions
    cfg->error_threshold = threshold;
    cfg->criteria        = goal;

    return WRFFT_SUCCESS;
}

WrFFTErrors wrfft_classify(ComplexData* data, WrFFTConfig* cfg) {
    if (!data||!cfg) return WRFFT_ERROR_INVALID_INPUT;
    const char* lib; const char* prec;
    classify_model(data, cfg, lib, prec);
    cfg->chosen_library = lib;
    cfg->chosen_precision = prec;
    return WRFFT_SUCCESS;
}

WrFFTErrors wrfft_plan(ComplexData* data, WrFFTConfig* cfg) {
    if (!data || !cfg) return WRFFT_ERROR_INVALID_INPUT;
    int nx = (int)cfg->data_size[0];
    int ny = (int)cfg->data_size[1];
    int n  = nx * ny;
    WrFFTErrors err = WRFFT_SUCCESS;
    const CufftPrecision prec = to_enum(cfg->chosen_precision);
    if (cfg->data_size[2] == 1) {
        // 1D
        Cufft1DContext* ctx = nullptr;
        err = cufft1d_plan(nx, &ctx, prec);
        if (err != WRFFT_SUCCESS) return err;
        cfg->internal_plan = ctx;
        cfg->gpu_input      = ctx->d_input;
    } else {
        // 2D
        Cufft2DContext* ctx = nullptr;
        err = cufft2d_plan(nx, ny, &ctx, prec);
        if (err != WRFFT_SUCCESS) return err;
        cfg->internal_plan = ctx;
        cfg->gpu_input      = ctx->d_input;
    }
    if (err != WRFFT_SUCCESS) return err;

    /* 2. Copy & convert host input -> device buffer */
    size_t bytes = (prec == CUFFT_PREC_SINGLE) ? sizeof(cufftComplex)      * n :
                   (prec == CUFFT_PREC_DOUBLE) ? sizeof(cufftDoubleComplex)* n :
                                                 sizeof(half2)             * n;

    switch (prec)
    {
        case CUFFT_PREC_SINGLE: {
            std::vector<cufftComplex> h(n);
            for (int i = 0; i < n; ++i) {
                h[i].x = static_cast<float>(data[i].real);
                h[i].y = static_cast<float>(data[i].imag);
            }
            CUDA_CHECK(cudaMemcpy(cfg->gpu_input, h.data(), bytes, cudaMemcpyHostToDevice));
            break;
        }
        case CUFFT_PREC_DOUBLE: {
            CUDA_CHECK(cudaMemcpy(cfg->gpu_input, data, bytes, cudaMemcpyHostToDevice));
            break;
        }
        case CUFFT_PREC_HALF: {
            std::vector<half2> h(n);
            for (int i = 0; i < n; ++i) {
                h[i].x = __double2half(data[i].real);
                h[i].y = __double2half(data[i].imag);
            }
            CUDA_CHECK(cudaMemcpy(cfg->gpu_input, h.data(), bytes, cudaMemcpyHostToDevice));
            break;
        }
    }
    return WRFFT_SUCCESS;
}

WrFFTErrors wrfft_execute(ComplexData* out, WrFFTConfig* cfg) {
    if (!out || !cfg || !cfg->internal_plan) return WRFFT_ERROR_INVALID_INPUT;
    // Cufft1DContext* ctx = static_cast<Cufft1DContext*>(cfg->internal_plan);
    WrFFTErrors err = WRFFT_SUCCESS;
    const CufftPrecision prec = to_enum(cfg->chosen_precision);
    if (cfg->data_size[2] == 1) {
        Cufft1DContext* ctx = static_cast<Cufft1DContext*>(cfg->internal_plan);
        err = cufft1d_execute(ctx, CUFFT_FORWARD);
        cfg->gpu_output = ctx->d_output;
    } else {
        Cufft2DContext* ctx = static_cast<Cufft2DContext*>(cfg->internal_plan);
        err = cufft2d_execute(ctx, CUFFT_FORWARD);
        cfg->gpu_output = ctx->d_output;
    }
    if (err != WRFFT_SUCCESS) {
        std::cerr << "  execute failed: " << err << "\n";
        (cfg->data_size[2] == 1) ? cufft1d_cleanup(static_cast<Cufft1DContext*>(cfg->internal_plan)) : cufft2d_cleanup(static_cast<Cufft2DContext*>(cfg->internal_plan));
    }

    /* 2. Copy & convert device -> out                                         */
    const int n     = cfg->data_size[0] * cfg->data_size[1];
    size_t    bytes = (prec == CUFFT_PREC_SINGLE) ? sizeof(cufftComplex)      * n :
                      (prec == CUFFT_PREC_DOUBLE) ? sizeof(cufftDoubleComplex)* n :
                                                    sizeof(half2)             * n;

    switch (prec)
    {
        case CUFFT_PREC_SINGLE: {
            std::vector<cufftComplex> h(n);
            CUDA_CHECK(cudaMemcpy(h.data(), cfg->gpu_output, bytes, cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) {
                out[i].real = h[i].x;
                out[i].imag = h[i].y;
            }
            break;
        }
        case CUFFT_PREC_DOUBLE: {
            CUDA_CHECK(cudaMemcpy(out, cfg->gpu_output, bytes, cudaMemcpyDeviceToHost));
            break;
        }
        case CUFFT_PREC_HALF: {
            std::vector<half2> h(n);
            CUDA_CHECK(cudaMemcpy(h.data(), cfg->gpu_output, bytes, cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) {
                out[i].real = static_cast<double>(h[i].x);
                out[i].imag = static_cast<double>(h[i].y);
            }
            break;
        }
    }
    return WRFFT_SUCCESS;
}

WrFFTErrors wrfft_finalize(WrFFTConfig* cfg) {
    if (!cfg || !cfg->internal_plan) return WRFFT_ERROR_INVALID_INPUT;

    WrFFTErrors err;
    if (cfg->data_size[2] == 1) {
        Cufft1DContext* ctx = static_cast<Cufft1DContext*>(cfg->internal_plan);
        err = cufft1d_cleanup(ctx);
    } else {
        Cufft2DContext* ctx = static_cast<Cufft2DContext*>(cfg->internal_plan);
        err = cufft2d_cleanup(ctx);
    }

    cfg->internal_plan = nullptr;
    cfg->gpu_input      = nullptr;
    cfg->gpu_output     = nullptr;
    return err;
}
