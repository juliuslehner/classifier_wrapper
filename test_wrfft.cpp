// test_wrfft.cpp
#include "wrfft.h"

#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    constexpr int N = 16;
    std::vector<ComplexData> in(N), out(N);

    // ───── delta-function: 1 + 0 i, rest 0
    in[0] = {1.0, 0.0};
    for (int i = 1; i < N; ++i) in[i] = {0.0, 0.0};

    WrFFTConfig cfg{};                                                // zero-initialise

    std::cout << "Initializing:\n";
    if (wrfft_initialize(4, 4, 1e-3, WRFFT_OPTIMIZE_SPEED, &cfg) != WRFFT_SUCCESS)
        return 1;
    std::cout << "Classifying: \n";
    if (wrfft_classify(in.data(), &cfg) != WRFFT_SUCCESS)
        return 2;

    std::cout << "Chosen back-end  : " << cfg.chosen_library
              << "\nChosen precision: " << cfg.chosen_precision << '\n';
    if (cfg.chosen_library != "cufft")
        cfg.chosen_library = "cufft"; // force to use cuFFT for this test
    if (wrfft_plan(in.data(), &cfg) != WRFFT_SUCCESS)
        return 3;
    if (wrfft_execute(out.data(), &cfg) != WRFFT_SUCCESS)
        return 4;
    if (wrfft_finalize(&cfg) != WRFFT_SUCCESS)
        return 5;

    std::cout << "FFT output:\n";
    std::cout << std::fixed << std::setprecision(7);
    for (auto& c : out) std::cout << '(' << c.real << ", " << c.imag << ") ";
    std::cout << '\n';
    return 0;
}
