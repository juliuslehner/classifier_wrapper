// test_wrfft.cpp
#include "wrfft.h"

#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    constexpr int N = 8192;
    std::vector<ComplexData> in(N);

    // ───── delta-function: 1 + 0 i, rest 0
    in[0] = {1.0, 0.0};
    for (int i = 1; i < N; ++i) in[i] = {0.0, 0.0};

    WrFFTConfig cfg{};                                                // zero-initialise

    std::cout << "Initializing:\n";
    if (wrfft_initialize(N, 1, 1e-3, OPTIMIZE_SPEED, &cfg) != WRFFT_SUCCESS)
        return 1;
    std::cout << "Classifying: \n";
    if (wrfft_classify(in.data(), &cfg) != WRFFT_SUCCESS)
        return 2;

    std::cout << "Chosen back-end  : " << cfg.chosen_library
              << "\nChosen precision: " << cfg.chosen_precision << '\n';
    if (wrfft_plan(&cfg) != WRFFT_SUCCESS){
        std::cout << "Plan failed\n";
        return 3;
    }
    if (wrfft_execute(in.data(), &cfg) != WRFFT_SUCCESS){
        std::cout << "Execution failed\n";
        return 4;
    }
    if (wrfft_finalize(&cfg) != WRFFT_SUCCESS){
        std::cout << "Finalization failed\n";
        return 5;
    }
    std::cout << "FFT output:\n";
    std::cout << std::fixed << std::setprecision(7);
    for (int i = 0; i < 8; ++i) {
        std::cout << '(' << in[i].x << ',' << in[i].y << ") ";
    }
    std::cout << '\n';
    return 0;
}
