#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cassert>
#include <pthread.h>

template <class T>
using matrix = std::vector<std::vector<T>>;

namespace ROW
{
    template <class T>
    struct multiply_args {
        std::shared_ptr<matrix<T>> m;
        std::shared_ptr<std::vector<T>> v;
        std::shared_ptr<std::vector<T>> out;
        size_t thread_idx;
        size_t num_threads;
    };

    pthread_mutex_t mutex;

    template <class T>
    void* multiply_worker(void* args) {
        multiply_args<T> *data = static_cast<multiply_args<T> *>(args);
        
        pthread_mutex_lock(&mutex);
        auto tread_idx = data->thread_idx++;
        // std::cout << tread_idx;
        pthread_mutex_unlock(&mutex);

        size_t M = data->m->size(), N = data->m->at(0).size();
        size_t step = (M + data->num_threads - 1) / data->num_threads; 
        for (size_t i = 0; i < data->num_threads; ++i) {
            for (size_t j = i * step; i == tread_idx && j < (i + 1) * step && j < M; ++j) {
                data->out->at(j) = 0;
                for (size_t e = 0; e < N; ++e) {
                    data->out->at(j)  += data->m->at(j)[e] * data->v->at(e);
                }
            }
        }
        
        pthread_exit(NULL); 
        return NULL;
    }

    template <class T>
    std::vector<T> multiply(std::shared_ptr<matrix<T>> m, std::shared_ptr<std::vector<T>> v, size_t MAX_THREAD=32) {
        assert(m->at(0).size() == v->size());
        size_t M = m->size(), N = m->at(0).size();

        auto out = std::make_shared<std::vector<T>>(std::vector<T>(M, 0));
        size_t num_threads = std::min(M, static_cast<size_t>(MAX_THREAD));
        multiply_args<T> args{m, v, out, 0, num_threads};

        pthread_t threads[num_threads];
        for (int i = 0; i < num_threads; ++i) {
            pthread_create(&threads[i], NULL, &multiply_worker<T>, (void*)&args);
        }
    
        for (int i = 0; i < num_threads; ++i) {
            pthread_join(threads[i], NULL);
        }

        return *out;
    }
}

namespace COL
{
    template <class T>
    struct multiply_args {
        std::shared_ptr<matrix<T>> m;
        std::shared_ptr<std::vector<T>> v;
        std::shared_ptr<matrix<T>> out;
        size_t thread_idx;
        size_t num_threads;
    };

    pthread_mutex_t mutex;

    template <class T>
    void* multiply_worker(void* args) {
        multiply_args<T> *data = static_cast<multiply_args<T> *>(args);
        
        pthread_mutex_lock(&mutex);
        auto tread_idx = data->thread_idx++;
        // std::cout << tread_idx;
        pthread_mutex_unlock(&mutex);

        size_t M = data->m->size(), N = data->m->at(0).size();
        size_t step = (N + data->num_threads - 1) / data->num_threads; 
        for (size_t i = 0; i < data->num_threads; ++i) {
            for (size_t j = i * step; i == tread_idx && j < (i + 1) * step && j < N; ++j) {
                for (size_t e = 0; e < M; ++e) {
                    data->out->at(e)[j]  = data->m->at(e)[j] * data->v->at(j);
                }
            }
        }
        
        pthread_exit(NULL); 
        return NULL;
    }

    template <class T>
    struct summation_args {
        std::shared_ptr<matrix<T>> m;
        std::shared_ptr<std::vector<T>> out;
        size_t thread_idx;
        size_t num_threads;
    };

    template <class T>
    void* summation_worker(void* args) {
        summation_args<T> *data = static_cast<summation_args<T> *>(args);
        
        pthread_mutex_lock(&mutex);
        auto tread_idx = data->thread_idx++;
        // std::cout << tread_idx;
        pthread_mutex_unlock(&mutex);

        size_t M = data->m->size(), N = data->m->at(0).size();
        size_t step = (M + data->num_threads - 1) / data->num_threads; 
        for (size_t i = 0; i < data->num_threads; ++i) {
            for (size_t j = i * step; i == tread_idx && j < (i + 1) * step && j < M; ++j) {
                data->out->at(j) = 0;
                for (size_t e = 0; e < N; ++e) {
                    data->out->at(j) += data->m->at(j)[e];
                }
            }
        }
        
        pthread_exit(NULL); 
        return NULL;
    }

    template <class T>
    std::vector<T> multiply(std::shared_ptr<matrix<T>> m, std::shared_ptr<std::vector<T>> v, size_t MAX_THREAD=32) {
        assert(m->at(0).size() == v->size());
        size_t M = m->size(), N = m->at(0).size();

        auto out = std::make_shared<matrix<T>>(std::vector<std::vector<T>>(M, std::vector<T>(N, 0)));
        size_t num_threads = std::min(M, static_cast<size_t>(MAX_THREAD));
        multiply_args<T> args{m, v, out, 0, num_threads};

        pthread_t threads[num_threads];
        for (int i = 0; i < num_threads; ++i) {
            pthread_create(&threads[i], NULL, &multiply_worker<T>, (void*)&args);
        }
    
        for (int i = 0; i < num_threads; ++i) {
            pthread_join(threads[i], NULL);
        }

        // std::cout << "\n\nC:\n";
        // for (int j = 0; j < M; ++j) {
        //     for (int i = 0; i < N; ++i) {
        //         std::cout << (*out)[j][i] << " ";
        //     }
        //     std::cout << "\n";
        // }

        auto out_vec = std::make_shared<std::vector<T>>(std::vector<T>(M, 0));
        summation_args<T> sargs{out, out_vec, 0, num_threads};
        for (int i = 0; i < num_threads; ++i) {
            pthread_create(&threads[i], NULL, &summation_worker<T>, (void*)&sargs);
        }
    
        for (int i = 0; i < num_threads; ++i) {
            pthread_join(threads[i], NULL);
        }

        return *out_vec;
    }
}

namespace BLOCK
{
    pthread_mutex_t mutex;

    template <class T>
    void* multiply_worker(void* args) {
        COL::multiply_args<T> *data = static_cast<COL::multiply_args<T> *>(args);
        
        pthread_mutex_lock(&mutex);
        auto tread_idx = data->thread_idx++;
        // std::cout << tread_idx;
        pthread_mutex_unlock(&mutex);

        size_t M = data->m->size(), N = data->m->at(0).size();
        size_t step_m = (M + data->num_threads - 1) / data->num_threads; 
        size_t step_n = (N + data->num_threads - 1) / data->num_threads; 
        for (size_t j = 0; j < data->num_threads; ++j) {
            for (size_t i = 0; i < data->num_threads && j == tread_idx; ++i) {
                for (size_t ii = i * step_m; ii < (i + 1) * step_m && ii < M; ++ii) {
                    for (size_t jj = j * step_n; jj < (j + 1) * step_n && jj < N; ++jj) {
                        data->out->at(ii)[j] += data->m->at(ii)[jj] * data->v->at(jj);
                    }
                }
            }
        }
        
        pthread_exit(NULL); 
        return NULL;
    }

    template <class T>
    std::vector<T> multiply(std::shared_ptr<matrix<T>> m, std::shared_ptr<std::vector<T>> v, size_t MAX_THREAD=32) {
        assert(m->at(0).size() == v->size());
        size_t M = m->size(), N = m->at(0).size();

        size_t num_threads = std::min(N, std::min(M, static_cast<size_t>(MAX_THREAD)));
        auto out = std::make_shared<matrix<T>>(std::vector<std::vector<T>>(M, std::vector<T>(num_threads, 0)));
        COL::multiply_args<T> args{m, v, out, 0, num_threads};

        pthread_t threads[num_threads];
        for (int i = 0; i < num_threads; ++i) {
            pthread_create(&threads[i], NULL, &multiply_worker<T>, (void*)&args);
        }
    
        for (int i = 0; i < num_threads; ++i) {
            pthread_join(threads[i], NULL);
        }

        // std::cout << "\n\nC:\n";
        // for (int j = 0; j < M; ++j) {
        //     for (int i = 0; i < num_threads; ++i) {
        //         std::cout << (*out)[j][i] << " ";
        //     }
        //     std::cout << "\n";
        // }

        auto out_vec = std::make_shared<std::vector<T>>(std::vector<T>(M, 0));
        COL::summation_args<T> sargs{out, out_vec, 0, num_threads};
        for (int i = 0; i < num_threads; ++i) {
            pthread_create(&threads[i], NULL, &COL::summation_worker<T>, (void*)&sargs);
        }
    
        for (int i = 0; i < num_threads; ++i) {
            pthread_join(threads[i], NULL);
        }

        return *out_vec;
    }
}

#endif  // MATRIX_MULTIPLY_H
