#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <fstream>

#include "qr_decomposition.hpp"

#define DEMONSTRATION_MODE

namespace RAND
{
    std::random_device dev;
    std::mt19937 rng(dev());
    double Rand(double fMin, double fMax) {
        std::uniform_int_distribution<std::mt19937::result_type> dist(fMin, fMax);
        return dist(rng);
    }

    std::default_random_engine re;
    double fRand(double fMin, double fMax) {
        std::uniform_real_distribution<double> unif(fMin,fMax);
        return unif(re);
    }

    template <class T>
    QR::matrix<T> generate_matrix(size_t M, size_t N, T fMin, T fMax);

    template <>
    QR::matrix<double> generate_matrix(size_t M, size_t N, double fMin, double fMax) {
        auto mat = QR::matrix<double>(M, std::vector<double>(N));
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                mat[i][j] = fRand(fMin, fMax);
            }
        }
        return mat;
    }
}

template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

#ifdef DEMONSTRATION_MODE

int main() {
    auto m = std::make_shared<QR::matrix<double>>(QR::matrix<double>{{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}});
    // auto m = std::make_shared<QR::matrix<double>>(QR::matrix<double>{{2, 3}, {2, 4}, {1, 1}});

    std::cout << "A (" << (*m).size() << " x " << (*m)[0].size() << "):\n";
    for (size_t j = 0; j < (*m)[0].size(); ++j) {
        for (size_t i = 0; i < (*m).size(); ++i) {
            std::cout << std::fixed << std::right;
            std::cout.fill(' ');
            std::cout.width(12);
            std::cout.precision(6);
            if (std::abs((*m)[i][j]) < EPSILON) {
                std::cout << "0 ";
                continue;
            } 
            std::cout << (*m)[i][j];
        }
        std::cout << "\n"; 
    }

    auto out = QR::decompose(m);
    auto Q = out.first, R = out.second;

    std::cout << "\nQ (" << Q[0].size() << " x " << Q.size() << "):\n";
    for (size_t j = 0; j < Q[0].size(); ++j) {
        for (size_t i = 0; i < Q.size(); ++i) {
            std::cout << std::fixed << std::right;
            std::cout.fill(' ');
            std::cout.width(12);
            std::cout.precision(6);
            if (std::abs(Q[i][j]) < EPSILON) {
                std::cout << "0 ";
                continue;
            } 
            std::cout << Q[i][j];
        }
        std::cout << "\n"; 
    }

    std::cout << "\nR (" << R.size() << " x " << R[0].size() << "):\n";
    for (size_t i = 0; i < R.size(); ++i) {
        for (size_t j = 0; j < R[0].size(); ++j) {
            std::cout << std::fixed << std::right;
            std::cout.fill(' ');
            std::cout.width(12);
            std::cout.precision(6);
            if (std::abs(R[i][j]) < EPSILON) {
                std::cout << "0 ";
                continue;
            } 
            std::cout << R[i][j];
        }
        std::cout << "\n"; 
    }

    return 0;
}

#endif  // DEMONSTRATION_MODE

#ifndef DEMONSTRATION_MODE

int main() {
    std::vector<size_t> threads = {1, 2, 4, 8, 16, 32};
    std::vector<size_t> mat_dim = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};

    std::ofstream fout;
    fout.open("qr_decomposition.csv");
    fout << "algorithm,type,treads,dim,time\n"; 

    for (size_t d : mat_dim) {

        size_t M = d, N = d;
        auto m = std::make_shared<QR::matrix<double>>(RAND::generate_matrix<double>(M, N, -100, 100));

        for (size_t t : threads) {
            std::cout << "t: " << t << " d: " << d << "\n";
            size_t elapsed = 0, repeats = 10;
            for (size_t tr = 0; tr < repeats; ++tr) {
                auto start = std::chrono::steady_clock::now();
                auto out = QR::decompose(m, t);
                auto Q = out.first, R = out.second;
                elapsed += since(start).count();
            }
            elapsed /= repeats;
            fout << "Gram–Schmidt,float,"<< t << ',' << d << ',' << elapsed << '\n';
        }
    }

    fout.close();
    return 0;
}

#endif  // DEMONSTRATION_MODE
