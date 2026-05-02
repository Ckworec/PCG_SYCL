#include "func.h"

using namespace std;

template <typename SolverFn>
static BenchmarkResult run_benchmark(const string& preconditioner,
                                     const string& device,
                                     vector<double>& x,
                                     vector<double>& b,
                                     int& number_iter,
                                     SolverFn&& solver_fn)
{
    struct timespec ts1, ts2;
    double full_time = 0.0;
    unsigned long long full_iterations = 0;
    char time_str[26];

    sycl::queue cpu_queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
    sycl::queue gpu_queue(nvidia_selector, sycl::property::queue::in_order());
    sycl::queue q = (device == "CPU") ? cpu_queue : gpu_queue;

    vector<double> x_test(b.size());
    BenchmarkResult result;
    result.preconditioner = preconditioner;
    result.device_name = q.get_device().get_info<sycl::info::device::name>();

    cout << "Using device: " << result.device_name << endl;

    auto now = chrono::system_clock::now();
    time_t start_time = chrono::system_clock::to_time_t(now);
    ctime_s(time_str, sizeof(time_str), &start_time);
    cout << "Start time: " << time_str;

    for (int i = 0; i < number_iter; ++i) {
        timespec_get(&ts1, TIME_UTC);
        try {
            full_iterations += solver_fn(q, x_test);
        } catch (const std::exception& e) {
            cerr << preconditioner << " failed: " << e.what() << endl;
            throw;
        }
        timespec_get(&ts2, TIME_UTC);
        full_time += (double(ts2.tv_sec) + double(ts2.tv_nsec) / 1000000000.0)
            - (double(ts1.tv_sec) + double(ts1.tv_nsec) / 1000000000.0);
    }

    x = x_test;

    now = chrono::system_clock::now();
    time_t end_time = chrono::system_clock::to_time_t(now);
    ctime_s(time_str, sizeof(time_str), &end_time);
    cout << "End time: " << time_str;

    result.average_time_sec = full_time / number_iter;
    result.iterations = static_cast<unsigned int>(full_iterations / number_iter);

    cout << device << " CG " << preconditioner << " average time for "
         << number_iter << " iterations: " << result.average_time_sec << endl << endl;

    return result;
}

template <typename SolverFn>
static BenchmarkResult run_host_benchmark(const string& preconditioner,
                                          const string& library,
                                          vector<double>& x,
                                          vector<double>& b,
                                          int& number_iter,
                                          SolverFn&& solver_fn)
{
    struct timespec ts1, ts2;
    double full_time = 0.0;
    unsigned long long full_iterations = 0;
    char time_str[26];

    sycl::queue cpu_queue(sycl::cpu_selector_v, sycl::property::queue::in_order());

    vector<double> x_test(b.size());
    BenchmarkResult result;
    result.preconditioner = preconditioner;
    result.device_name = cpu_queue.get_device().get_info<sycl::info::device::name>();
    result.library = library;

    cout << "Using device: " << result.device_name << " [" << library << "]" << endl;

    auto now = chrono::system_clock::now();
    time_t start_time = chrono::system_clock::to_time_t(now);
    ctime_s(time_str, sizeof(time_str), &start_time);
    cout << "Start time: " << time_str;

    for (int i = 0; i < number_iter; ++i) {
        timespec_get(&ts1, TIME_UTC);
        try {
            full_iterations += solver_fn(x_test);
        } catch (const std::exception& e) {
            cerr << preconditioner << " [" << library << "] failed: " << e.what() << endl;
            throw;
        }
        timespec_get(&ts2, TIME_UTC);
        full_time += (double(ts2.tv_sec) + double(ts2.tv_nsec) / 1000000000.0)
            - (double(ts1.tv_sec) + double(ts1.tv_nsec) / 1000000000.0);
    }

    x = x_test;

    now = chrono::system_clock::now();
    time_t end_time = chrono::system_clock::to_time_t(now);
    ctime_s(time_str, sizeof(time_str), &end_time);
    cout << "End time: " << time_str;

    result.average_time_sec = full_time / number_iter;
    result.iterations = static_cast<unsigned int>(full_iterations / number_iter);

    cout << library << " CG " << preconditioner << " average time for "
         << number_iter << " iterations: " << result.average_time_sec << endl << endl;

    return result;
}

BenchmarkResult test_CG_SYCL_jacobi(CSR_matrix<double>& mat,
                                    vector<double>& x,
                                    vector<double>& b,
                                    string& device,
                                    int& number_iter)
{
    BenchmarkResult result = run_benchmark("Jacobi", device, x, b, number_iter,
        [&](sycl::queue& q, vector<double>& x_test) {
            return CG_SYCL_jacobi(mat, x_test, b, q);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    return result;
}

BenchmarkResult test_CG_SYCL_plain(CSR_matrix<double>& mat,
                                   vector<double>& x,
                                   vector<double>& b,
                                   string& device,
                                   int& number_iter)
{
    BenchmarkResult result = run_benchmark("CG", device, x, b, number_iter,
        [&](sycl::queue& q, vector<double>& x_test) {
            return CG_SYCL_plain(mat, x_test, b, q);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    return result;
}

BenchmarkResult test_CG_SYCL_IC0(CSR_matrix<double>& mat,
                                 vector<double>& x,
                                 vector<double>& b,
                                 string& device,
                                 int& number_iter)
{
    IC0Preconditioner ic0;
    ic0_factor(mat.take_rows(), mat.row_ptr_ref(), mat.col_ind_ref(), mat.val_ref(), ic0);

    BenchmarkResult result = run_benchmark("IC0", device, x, b, number_iter,
        [&](sycl::queue& q, vector<double>& x_test) {
            return CG_SYCL_IC0(mat, x_test, b, q, ic0);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    return result;
}

BenchmarkResult test_CG_SYCL_block_jacobi(CSR_matrix<double>& mat,
                                          vector<double>& x,
                                          vector<double>& b,
                                          string& device,
                                          int& number_iter,
                                          size_t block_size)
{
    BenchmarkResult result = run_benchmark("Block Jacobi", device, x, b, number_iter,
        [&](sycl::queue& q, vector<double>& x_test) {
            return CG_SYCL_block_jacobi(mat, x_test, b, q, block_size);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    result.block_size = block_size;
    return result;
}

BenchmarkResult test_CG_SYCL_SPAI(CSR_matrix<double>& mat,
                                  vector<double>& x,
                                  vector<double>& b,
                                  string& device,
                                  int& number_iter)
{
    BenchmarkResult result = run_benchmark("SPAI", device, x, b, number_iter,
        [&](sycl::queue& q, vector<double>& x_test) {
            return CG_SYCL_SPAI(mat, x_test, b, q);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    return result;
}

BenchmarkResult test_CG_SYCL_chebyshev(CSR_matrix<double>& mat,
                                       vector<double>& x,
                                       vector<double>& b,
                                       string& device,
                                       int& number_iter,
                                       size_t degree)
{
    size_t used_degree = degree;
    BenchmarkResult result = run_benchmark("Chebyshev", device, x, b, number_iter,
        [&](sycl::queue& q, vector<double>& x_test) {
            return CG_SYCL_chebyshev(mat, x_test, b, q, degree, &used_degree);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    result.polynomial_degree = used_degree;
    return result;
}

BenchmarkResult test_CG_SYCL_chebyshev_adaptive(CSR_matrix<double>& mat,
                                                vector<double>& x,
                                                vector<double>& b,
                                                string& device,
                                                int& number_iter,
                                                size_t degree)
{
    size_t used_degree = degree;
    BenchmarkResult result = run_benchmark("Chebyshev (Adaptive)", device, x, b, number_iter,
        [&](sycl::queue& q, vector<double>& x_test) {
            return CG_SYCL_chebyshev_adaptive(mat, x_test, b, q, degree, &used_degree);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    result.polynomial_degree = used_degree;
    return result;
}

BenchmarkResult test_CG_MKL_jacobi(CSR_matrix<double>& mat,
                                   vector<double>& x,
                                   vector<double>& b,
                                   int& number_iter)
{
    BenchmarkResult result = run_host_benchmark("Jacobi", "MKL", x, b, number_iter,
        [&](vector<double>& x_test) {
            return CG_MKL_jacobi(mat, x_test, b);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    return result;
}

BenchmarkResult test_CG_MKL_plain(CSR_matrix<double>& mat,
                                  vector<double>& x,
                                  vector<double>& b,
                                  int& number_iter)
{
    BenchmarkResult result = run_host_benchmark("CG", "MKL", x, b, number_iter,
        [&](vector<double>& x_test) {
            return CG_MKL_plain(mat, x_test, b);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    return result;
}

BenchmarkResult test_CG_MKL_SPAI(CSR_matrix<double>& mat,
                                 vector<double>& x,
                                 vector<double>& b,
                                 int& number_iter)
{
    BenchmarkResult result = run_host_benchmark("SPAI", "MKL", x, b, number_iter,
        [&](vector<double>& x_test) {
            return CG_MKL_SPAI(mat, x_test, b);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    return result;
}

BenchmarkResult test_CG_MKL_block_jacobi(CSR_matrix<double>& mat,
                                         vector<double>& x,
                                         vector<double>& b,
                                         int& number_iter,
                                         size_t block_size)
{
    BenchmarkResult result = run_host_benchmark("Block Jacobi", "MKL", x, b, number_iter,
        [&](vector<double>& x_test) {
            return CG_MKL_block_jacobi(mat, x_test, b, block_size);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    result.block_size = block_size;
    return result;
}

BenchmarkResult test_CG_MKL_chebyshev(CSR_matrix<double>& mat,
                                      vector<double>& x,
                                      vector<double>& b,
                                      int& number_iter,
                                      size_t degree)
{
    size_t used_degree = degree;
    BenchmarkResult result = run_host_benchmark("Chebyshev", "MKL", x, b, number_iter,
        [&](vector<double>& x_test) {
            return CG_MKL_chebyshev(mat, x_test, b, degree, &used_degree);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    result.polynomial_degree = used_degree;
    return result;
}

BenchmarkResult test_CG_MKL_IC0(CSR_matrix<double>& mat,
                                vector<double>& x,
                                vector<double>& b,
                                int& number_iter)
{
    IC0Preconditioner ic0;
    ic0_factor(mat.take_rows(), mat.row_ptr_ref(), mat.col_ind_ref(), mat.val_ref(), ic0);

    BenchmarkResult result = run_host_benchmark("IC0", "MKL", x, b, number_iter,
        [&](vector<double>& x_test) {
            return CG_MKL_IC0(mat, x_test, b, ic0);
        });
    result.matrix_rows = mat.take_rows();
    result.matrix_cols = mat.take_cols();
    result.matrix_nnz = mat.take_nnz();
    return result;
}
