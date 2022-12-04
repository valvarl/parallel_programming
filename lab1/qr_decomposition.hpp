#ifndef QR_DECOMPOSITION_H
#define QR_DECOMPOSITION_H

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <pthread.h>

#define EPSILON 1e-10

namespace QR
{
    template <class T>
    using matrix = std::vector<std::vector<T>>;

    template <class T>
    struct decompose_args {
        std::shared_ptr<matrix<T>> A;
        std::shared_ptr<matrix<T>> V;
        std::shared_ptr<matrix<T>> Q;
        std::shared_ptr<matrix<T>> R;
        size_t u_idx;
        size_t thread_idx;
        size_t num_threads;
    };

    template <class T>
    inline std::vector<T> proj_u(const matrix<T>& A, const matrix<T>& Q, size_t u, size_t a) {
        T s1 = 0, s2 = 0, proj;
        std::vector<T> out(A.size());
        for (size_t i = 0; i < A.size(); ++i) {
            s1 += Q[u][i] * A[i][a];
            s2 += Q[u][i] * Q[u][i];
        }
        proj = s1 / s2;
        for (size_t i = 0; i < A.size(); ++i) {
            out[i] = proj * Q[u][i];
        }
        return out;
    }

    template <class T>
    inline void norm(std::vector<T> *u) {
        T s = 0;
        for (size_t i = 0; i < u->size(); ++i) {
            s += u->at(i) * u->at(i);
        }
        if (abs(s) > EPSILON) {
            s = std::sqrt(s);
            for (size_t i = 0; i < u->size(); ++i) {
                u->at(i) /= s;
            }
        }
    }

    pthread_mutex_t mutex;

    template <class T>
    void* decompose_worker(void* args) {
        decompose_args<T> *data = static_cast<decompose_args<T> *>(args);
        
        pthread_mutex_lock(&mutex);
        auto tread_idx = data->thread_idx++;
        pthread_mutex_unlock(&mutex);

        size_t M = data->A->size(), N = data->A->at(0).size();
        size_t step = (data->u_idx + data->num_threads) / data->num_threads;
        for (size_t i = 0; i < data->num_threads; ++i) {
            for (size_t j = i * step; i == tread_idx && j < (i + 1) * step && j < data->u_idx; ++j) {
                // std::cout << "u :" << k << " a: " << row << '\n';
                auto proj = proj_u(*(data->A), *(data->Q), j, data->u_idx);
                data->V->at(j) = proj;
            }
        }
        
        pthread_exit(NULL); 
        return NULL;
    }

    template <class T>
    void* summation_worker(void* args) {
        decompose_args<T> *data = static_cast<decompose_args<T> *>(args);
        
        pthread_mutex_lock(&mutex);
        auto tread_idx = data->thread_idx++;
        pthread_mutex_unlock(&mutex);

        size_t M = data->A->size(), N = data->A->at(0).size();
        size_t step = (M + data->num_threads - 1) / data->num_threads;
        for (size_t i = 0; i < data->num_threads; ++i) {
            for (size_t j = i * step; i == tread_idx && j < (i + 1) * step && j < M; ++j) {
                // std::cout << "u :" << k << " a: " << row << '\n';
                for (size_t e = 0; e < data->u_idx; ++e) {
                    data->Q->at(data->u_idx)[j] -= data->V->at(e)[j];
                }
                data->Q->at(data->u_idx)[j] += data->A->at(j)[data->u_idx];
            }
        }
        
        pthread_exit(NULL); 
        return NULL;
    }

    template <class T>
    void* normalization_worker(void* args) {
        decompose_args<T> *data = static_cast<decompose_args<T> *>(args);
        
        pthread_mutex_lock(&mutex);
        auto tread_idx = data->thread_idx++;
        pthread_mutex_unlock(&mutex);

        size_t N = data->Q->size(), M = data->Q->at(0).size();
        size_t step = (N + data->num_threads - 1) / data->num_threads;
        for (size_t i = 0; i < data->num_threads; ++i) {
            for (size_t j = i * step; i == tread_idx && j < (i + 1) * step && j < N; ++j) {
                norm(&data->Q->at(j));
            }
        }
        
        pthread_exit(NULL); 
        return NULL;
    }

    template <class T>
    void* mat_multiply_worker(void* args) {
        decompose_args<T> *data = static_cast<decompose_args<T> *>(args);
        
        pthread_mutex_lock(&mutex);
        auto tread_idx = data->thread_idx++;
        pthread_mutex_unlock(&mutex);

        size_t M = data->Q->size(), N = data->Q->at(0).size();
        size_t step = (M + data->num_threads - 1) / data->num_threads; 
        for (size_t i = 0; i < data->num_threads; ++i) {
            for (size_t j = i * step; i == tread_idx && j < (i + 1) * step && j < M; ++j) {
                for (size_t k = 0; k < N; ++k) {
                    // std::cout << "j: " << j << " k: " << k << "\n";
                    for (size_t e = 0; e < data->A->at(0).size(); ++e) {
                        data->R->at(j)[e] += data->Q->at(j)[k] * data->A->at(k)[e];
                    }
                }
            }
        }
        
        pthread_exit(NULL); 
        return NULL;
    }

    template <class T = double>
    auto decompose(std::shared_ptr<matrix<T>> m, size_t MAX_THREAD=32) {
        size_t M = m->size(), N = m->at(0).size();

        auto V = std::make_shared<matrix<T>>(std::vector<std::vector<T>>(N));
        auto Q = std::make_shared<matrix<T>>(std::vector<std::vector<T>>(N, std::vector<T>(M, 0)));
        auto R = std::make_shared<matrix<T>>(std::vector<std::vector<T>>(N, std::vector<T>(N, 0)));
        
        decompose_args<T> args{m, V, Q, R, 0, 0, 0};

        pthread_t threads[MAX_THREAD];
        int u = 0;
        for (; u < N; ++u) {
            args.u_idx = u;
            args.thread_idx = 0;
            args.num_threads = std::min(N, static_cast<size_t>(MAX_THREAD));
            for (int i = 0; i < args.num_threads; ++i) {
                pthread_create(&threads[i], NULL, &decompose_worker<T>, (void*)&args);
            }

            for (int i = 0; i < args.num_threads; ++i) {
                pthread_join(threads[i], NULL);
            }

            args.thread_idx = 0;
            args.num_threads = std::min(M, static_cast<size_t>(MAX_THREAD));
             for (int i = 0; i < args.num_threads; ++i) {
                pthread_create(&threads[i], NULL, &summation_worker<T>, (void*)&args);
            }

            for (int i = 0; i < args.num_threads; ++i) {
                pthread_join(threads[i], NULL);
            }
            T s = 0;
            for (T e : args.Q->at(args.u_idx)) {
                s += e;
            }
            if (abs(s) < EPSILON) {
                N = u + 1;
                Q->resize(u + 1);
                R->resize(u + 1);
                break;
            }
        }

        args.thread_idx = 0;
        args.num_threads = std::min(N, static_cast<size_t>(MAX_THREAD));
        for (int i = 0; i < args.num_threads; ++i) {
            pthread_create(&threads[i], NULL, &normalization_worker<T>, (void*)&args);
        }

        for (int i = 0; i < args.num_threads; ++i) {
            pthread_join(threads[i], NULL);
        }

        args.thread_idx = 0;
        args.num_threads = std::min(N, static_cast<size_t>(MAX_THREAD));
        for (int i = 0; i < args.num_threads; ++i) {
            pthread_create(&threads[i], NULL, &mat_multiply_worker<T>, (void*)&args);
        }

        for (int i = 0; i < args.num_threads; ++i) {
            pthread_join(threads[i], NULL);
        }

        return std::make_pair(*Q, *R);
    }
}

#endif  // QR_DECOMPOSITION_H
 