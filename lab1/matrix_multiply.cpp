#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <fstream>

#include "matrix_multiply.hpp"

#define EPSILON 1e-10
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
    matrix<T> generate_matrix(size_t M, size_t N, T fMin, T fMax);

    template <>
    matrix<int> generate_matrix(size_t M, size_t N, int Min, int Max) {
        auto mat = matrix<int>(M, std::vector<int>(N));
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                mat[i][j] = Rand(Min, Max);
            }
        }
        return mat;
    }

    template <>
    matrix<double> generate_matrix(size_t M, size_t N, double fMin, double fMax) {
        auto mat = matrix<double>(M, std::vector<double>(N));
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                mat[i][j] = fRand(fMin, fMax);
            }
        }
        return mat;
    }

    template <class T>
    std::vector<T> generate_vector(size_t N, T fMin, T fMax);

    template <>
    std::vector<int> generate_vector(size_t N, int Min, int Max) {
        auto vec = std::vector<int>(N);
        for (size_t j = 0; j < N; ++j) {
            vec[j] = fRand(Min, Max);
        }
        return vec;
    }

    template <>
    std::vector<double> generate_vector(size_t N, double fMin, double fMax) {
        auto vec = std::vector<double>(N);
        for (size_t j = 0; j < N; ++j) {
            vec[j] = fRand(fMin, fMax);
        }
        return vec;
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
    auto m = std::make_shared<matrix<int>>(matrix<int>{{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}});
    auto v = std::make_shared<std::vector<int>>(std::vector<int>{-14, -70, 35});

    std::cout << "A (" << m->size() << " x " << m->at(0).size() << "):\n";
    for (size_t i = 0; i < (*m).size(); ++i) {
        for (size_t j = 0; j < (*m)[0].size(); ++j) {
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

    std::cout << "\nb (" << v->size() << " x 1):\n";
    for (auto i : *v) {
        std::cout << std::fixed << std::right;
        std::cout.fill(' ');
        std::cout.width(10);
        std::cout.precision(6);
        std::cout << i << " ";
    }

    size_t num_treads = 8;
    auto out = ROW::multiply<int>(m, v, num_treads);  // ROW, COL, BLOCK

    std::cout << "\n\nX (" << out.size() << " x 1):\n";
    for (auto i : out) {
        std::cout << std::fixed << std::right;
        std::cout.fill(' ');
        std::cout.width(10);
        std::cout.precision(6);
        std::cout << i << " ";
    }
}

#endif  // DEMONSTRATION_MODE

#ifndef DEMONSTRATION_MODE

int main() {
    std::vector<size_t> threads = {1, 2, 4, 8, 16, 32};
    std::vector<size_t> mat_dim = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};

    std::ofstream fout;
    fout.open("matrix_multiply_float.csv");
    fout << "algorithm,type,treads,dim,time\n"; 

    for (size_t d : mat_dim) {

        size_t M = d, N = d;
        auto m = std::make_shared<matrix<double>>(RAND::generate_matrix<double>(M, N, -100, 100));
        auto v = std::make_shared<std::vector<double>>(RAND::generate_vector<double>(N, -100, 100));

        for (size_t t : threads) {
            std::cout << "t: " << t << " d: " << d << "\n";
            size_t elapsed = 0, repeats = 10;
            for (size_t tr = 0; tr < repeats; ++tr) {
                auto start = std::chrono::steady_clock::now();
                auto out = ROW::multiply<double>(m, v, t);
                elapsed += since(start).count();
            }

            elapsed /= repeats;
            fout << "row,int,"<< t << ',' << d << ',' << elapsed << '\n';

            elapsed = 0;
            for (size_t tr = 0; tr < repeats; ++tr) {
                auto start = std::chrono::steady_clock::now();
                auto out = COL::multiply<double>(m, v, t);
                elapsed += since(start).count();
            }

            elapsed /= repeats;
            fout << "col,int,"<< t << ',' << d << ',' << elapsed << '\n';

            elapsed = 0;
            for (size_t tr = 0; tr < repeats; ++tr) {
                auto start = std::chrono::steady_clock::now();
                auto out = BLOCK::multiply<double>(m, v, t);
                elapsed += since(start).count();
            }

            elapsed /= repeats;
            fout << "block,int,"<< t << ',' << d << ',' << elapsed << '\n';
        }
    }

    fout.close();
    return 0;
}

#endif  // DEMONSTRATION_MODE
