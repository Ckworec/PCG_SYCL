#pragma once

#include "class.h"

#define eps 1.e-13

// --------------- Help functions ---------------------

struct BenchmarkResult {
    std::string preconditioner;
    std::string device_name;
    std::string library = "Custom";
    size_t matrix_rows = 0;
    size_t matrix_cols = 0;
    size_t matrix_nnz = 0;
    double average_time_sec = 0.0;
    unsigned int iterations = 0;
    double error_pct = 0.0;
    size_t block_size = 0;
    size_t polynomial_degree = 0;
};

// Diagonal-shifted IC0 factorisation (level-scheduled) — port of the
// standalone implementation in ic0_sycl/ic0_cg_sycl.cpp.
struct IC0Preconditioner {
    std::vector<size_t> row_ptr;       // copy of A's row_ptr (kept so the host apply is self-contained)
    std::vector<size_t> col_idx;       // copy of A's col_ind
    std::vector<size_t> diag_pos;
    std::vector<double> L_vals;
    std::vector<double> diag;
    std::vector<size_t> forward_level_ptr;
    std::vector<size_t> forward_rows;
    std::vector<size_t> backward_level_ptr;
    std::vector<size_t> backward_rows;
    std::vector<size_t> upper_ptr;
    std::vector<size_t> upper_row_idx;
    std::vector<size_t> upper_pos;
};

inline size_t choose_host_parallel_workers(size_t work_size, size_t min_chunk_size)
{
    if (work_size == 0 || work_size < min_chunk_size) {
        return 1;
    }

    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t max_workers = (work_size + min_chunk_size - 1) / min_chunk_size;
    return std::max<size_t>(1, std::min<size_t>(hw_threads, max_workers));
}

template <typename Func>
inline void parallel_for_host(size_t begin, size_t end, size_t min_chunk_size, Func&& fn)
{
    if (end <= begin) {
        return;
    }

    const size_t work_size = end - begin;
    const size_t worker_count = choose_host_parallel_workers(work_size, min_chunk_size);
    if (worker_count <= 1) {
        for (size_t idx = begin; idx < end; ++idx) {
            fn(idx);
        }
        return;
    }

    const size_t chunk_size = (work_size + worker_count - 1) / worker_count;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
        const size_t chunk_begin = begin + worker_id * chunk_size;
        const size_t chunk_end = std::min(end, chunk_begin + chunk_size);
        if (chunk_begin >= chunk_end) {
            continue;
        }

        workers.emplace_back([chunk_begin, chunk_end, &fn]() {
            for (size_t idx = chunk_begin; idx < chunk_end; ++idx) {
                fn(idx);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
}

template <typename T, typename Func>
inline T parallel_sum_host(size_t begin, size_t end, size_t min_chunk_size, T init, Func&& fn)
{
    if (end <= begin) {
        return init;
    }

    const size_t work_size = end - begin;
    const size_t worker_count = choose_host_parallel_workers(work_size, min_chunk_size);
    if (worker_count <= 1) {
        T result = init;
        for (size_t idx = begin; idx < end; ++idx) {
            result += fn(idx);
        }
        return result;
    }

    const size_t chunk_size = (work_size + worker_count - 1) / worker_count;
    std::vector<T> partial(worker_count, T{});
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
        const size_t chunk_begin = begin + worker_id * chunk_size;
        const size_t chunk_end = std::min(end, chunk_begin + chunk_size);
        if (chunk_begin >= chunk_end) {
            continue;
        }

        workers.emplace_back([chunk_begin, chunk_end, worker_id, &partial, &fn]() {
            T local{};
            for (size_t idx = chunk_begin; idx < chunk_end; ++idx) {
                local += fn(idx);
            }
            partial[worker_id] = local;
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    T result = init;
    for (const T& value : partial) {
        result += value;
    }
    return result;
}

template <typename T, typename Func>
inline T parallel_max_host(size_t begin, size_t end, size_t min_chunk_size, T init, Func&& fn)
{
    if (end <= begin) {
        return init;
    }

    const size_t work_size = end - begin;
    const size_t worker_count = choose_host_parallel_workers(work_size, min_chunk_size);
    if (worker_count <= 1) {
        T result = init;
        for (size_t idx = begin; idx < end; ++idx) {
            result = std::max(result, fn(idx));
        }
        return result;
    }

    const size_t chunk_size = (work_size + worker_count - 1) / worker_count;
    std::vector<T> partial(worker_count, init);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
        const size_t chunk_begin = begin + worker_id * chunk_size;
        const size_t chunk_end = std::min(end, chunk_begin + chunk_size);
        if (chunk_begin >= chunk_end) {
            continue;
        }

        workers.emplace_back([chunk_begin, chunk_end, worker_id, &partial, &fn]() {
            T local = partial[worker_id];
            for (size_t idx = chunk_begin; idx < chunk_end; ++idx) {
                local = std::max(local, fn(idx));
            }
            partial[worker_id] = local;
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    T result = init;
    for (const T& value : partial) {
        result = std::max(result, value);
    }
    return result;
}

void read_vec(std::vector<double>& vec,
            std::string& name_file);

BenchmarkResult test_CG_SYCL_plain(CSR_matrix<double>& mat,
                                   std::vector<double>& x,
                                   std::vector<double>& b,
                                   std::string& device,
                                   int& number_iter);

BenchmarkResult test_CG_SYCL_jacobi(CSR_matrix<double>& mat, 
                                    std::vector<double>& x, 
                                    std::vector<double>& b, 
                                    std::string& device,
                                    int& number_iter);

BenchmarkResult test_CG_SYCL_IC0(CSR_matrix<double>& mat,
                                 std::vector<double>& x,
                                 std::vector<double>& b,
                                 std::string& device,
                                 int& number_iter);

BenchmarkResult test_CG_SYCL_block_jacobi(CSR_matrix<double>& mat,
                                          std::vector<double>& x,
                                          std::vector<double>& b,
                                          std::string& device,
                                          int& number_iter,
                                          size_t block_size = 16);

BenchmarkResult test_CG_SYCL_chebyshev_adaptive(CSR_matrix<double>& mat,
                                                std::vector<double>& x,
                                                std::vector<double>& b,
                                                std::string& device,
                                                int& number_iter,
                                                size_t degree = 4);

BenchmarkResult test_CG_MKL_jacobi(CSR_matrix<double>& mat,
                                   std::vector<double>& x,
                                   std::vector<double>& b,
                                   int& number_iter);

BenchmarkResult test_CG_MKL_plain(CSR_matrix<double>& mat,
                                  std::vector<double>& x,
                                  std::vector<double>& b,
                                  int& number_iter);

BenchmarkResult test_CG_MKL_block_jacobi(CSR_matrix<double>& mat,
                                         std::vector<double>& x,
                                         std::vector<double>& b,
                                         int& number_iter,
                                         size_t block_size = 16);

BenchmarkResult test_CG_MKL_chebyshev(CSR_matrix<double>& mat,
                                      std::vector<double>& x,
                                      std::vector<double>& b,
                                      int& number_iter,
                                      size_t degree = 4);

BenchmarkResult test_CG_MKL_IC0(CSR_matrix<double>& mat,
                                std::vector<double>& x,
                                std::vector<double>& b,
                                int& number_iter);

void append_benchmark_to_xlsx(const std::string& xlsx_path,
                              const std::string& matrix_name,
                              const BenchmarkResult& result);

double check_result(std::vector<double> first,
                std::vector<double> second);

double check_result(std::vector<double> x,
                    std::vector<double> b,
                    CSR_matrix<double>& mat);

// --------------- SYCL NVIDIA ------------------------

auto nvidia_selector = [](const sycl::device& dev) {
    const std::string name = dev.get_info<sycl::info::device::name>();
    if (name.find("NVIDIA") != std::string::npos) {
        return 1;
    }
    return -1;
};

static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const& e : e_list) {
        try {
            std::rethrow_exception(e);
        }
        catch (std::exception const& e) {
#if _DEBUG
            std::cout << "Failure" << std::endl;
#endif
            std::terminate();
        }
    }
};

void copy_from_nvidia(sycl::queue& q, 
                        sycl::buffer<double>& a_buf, 
                        double& a);

void copy_from_nvidia(sycl::queue& q, 
                        sycl::buffer<double>& a_buf, 
                        std::vector<double>& a);

// ----------------- Preconditioners --------------------

void compute_jacobi_preconditioner_buf(sycl::buffer<size_t>& row_ptr_buf, 
                                        sycl::buffer<size_t>& diag_pos_buf,
                                        sycl::buffer<double>& val_buf, 
                                        sycl::buffer<double>& M_inv_buf,
                                        size_t n, sycl::queue& q);

void build_diag_positions(const std::vector<size_t>& row_ptr,
                          const std::vector<size_t>& col_ind,
                          std::vector<size_t>& diag_pos);

void apply_diagonal_preconditioner(sycl::queue& q,
                                   sycl::buffer<double>& M_inv_buf,
                                   sycl::buffer<double>& r_buf,
                                   sycl::buffer<double>& z_buf,
                                   size_t n);

void compute_block_jacobi_preconditioner(const std::vector<size_t>& row_ptr,
                                         const std::vector<size_t>& col_ind,
                                         const std::vector<double>& vals,
                                         size_t n,
                                         size_t block_size,
                                         std::vector<double>& block_inv);

void compute_block_jacobi_preconditioner_device(sycl::queue& q,
                                                sycl::buffer<size_t>& row_ptr_buf,
                                                sycl::buffer<size_t>& col_ind_buf,
                                                sycl::buffer<double>& val_buf,
                                                sycl::buffer<double>& block_inv_buf,
                                                size_t n,
                                                size_t block_size);

void apply_block_jacobi_preconditioner(sycl::queue& q,
                                       sycl::buffer<double>& block_inv_buf,
                                       sycl::buffer<double>& r_buf,
                                       sycl::buffer<double>& z_buf,
                                       size_t n,
                                       size_t block_size);

void build_chebyshev_steps(double lambda_min_estimate,
                           double lambda_max_estimate,
                           size_t degree,
                           std::vector<double>& steps);

void apply_chebyshev_preconditioner_device(sycl::queue& q,
                                           sycl::buffer<size_t>& row_ptr_buf,
                                           sycl::buffer<size_t>& col_ind_buf,
                                           sycl::buffer<double>& val_buf,
                                           sycl::buffer<double>& M_inv_buf,
                                           sycl::buffer<double>& r_buf,
                                           sycl::buffer<double>& t_buf,
                                           sycl::buffer<double>& tmp_buf,
                                           sycl::buffer<double>& z_buf,
                                           size_t n,
                                           const std::vector<double>& steps,
                                           bool initialize_from_r = true);

// 32-bit-index overload — same semantics, halves col_ind bandwidth on GPU.
void apply_chebyshev_preconditioner_device(sycl::queue& q,
                                           sycl::buffer<uint32_t>& row_ptr_buf,
                                           sycl::buffer<uint32_t>& col_ind_buf,
                                           sycl::buffer<double>& val_buf,
                                           sycl::buffer<double>& M_inv_buf,
                                           sycl::buffer<double>& r_buf,
                                           sycl::buffer<double>& t_buf,
                                           sycl::buffer<double>& tmp_buf,
                                           sycl::buffer<double>& z_buf,
                                           size_t n,
                                           const std::vector<double>& steps,
                                           bool initialize_from_r = true);

void ic0_factor(size_t n,
                const std::vector<size_t>& row_ptr,
                const std::vector<size_t>& col_idx,
                const std::vector<double>& vals,
                IC0Preconditioner& ic0);

// IC0 application on the host — writes dst = M^{-1} src. y is scratch for
// the forward pass / Jacobi ping-pong. Mode is selected via env var
// IC0_JACOBI_SWEEPS: 0 -> exact level-scheduled solve, N>=1 -> N Jacobi
// sweeps in each direction (default 2).
void applyIC0_preconditioner_host(const IC0Preconditioner& ic0,
                                  const std::vector<double>& src,
                                  std::vector<double>& y,
                                  std::vector<double>& dst);

// SYCL Jacobi-sweep IC0 apply for the GPU. Each sweep is one bulk
// parallel_for over n rows; ns total sweeps in each direction. y_buf and
// y2_buf alternate as ping-pong scratch; final result lands in dst_buf.
void applyIC0_preconditioner_jacobi_device(sycl::queue& q,
                                           sycl::buffer<size_t>& row_ptr_buf,
                                           sycl::buffer<size_t>& col_idx_buf,
                                           sycl::buffer<double>& L_buf,
                                           sycl::buffer<double>& diag_buf,
                                           sycl::buffer<size_t>& diag_pos_buf,
                                           sycl::buffer<size_t>& upper_ptr_buf,
                                           sycl::buffer<size_t>& upper_row_idx_buf,
                                           sycl::buffer<size_t>& upper_pos_buf,
                                           sycl::buffer<double>& src_buf,
                                           sycl::buffer<double>& y_buf,
                                           sycl::buffer<double>& y2_buf,
                                           sycl::buffer<double>& dst_buf,
                                           size_t n,
                                           int ns);

// -------------- Solvers -----------------

unsigned int CG_SYCL_plain(CSR_matrix<double>& mat,
                           std::vector<double>& x,
                           std::vector<double>& b,
                           sycl::queue& q);

unsigned int CG_SYCL_jacobi(CSR_matrix<double>& mat,
                            std::vector<double>& x, 
                            std::vector<double>& b, 
                            sycl::queue& q);

unsigned int CG_SYCL_IC0(CSR_matrix<double>& mat,
                         std::vector<double>& x,
                         std::vector<double>& b,
                         sycl::queue& q);

unsigned int CG_SYCL_IC0(CSR_matrix<double>& mat,
                         std::vector<double>& x,
                         std::vector<double>& b,
                         sycl::queue& q,
                         const IC0Preconditioner& ic0);

unsigned int CG_SYCL_block_jacobi(CSR_matrix<double>& mat,
                                  std::vector<double>& x,
                                  std::vector<double>& b,
                                  sycl::queue& q,
                                  size_t block_size = 16);

unsigned int CG_SYCL_chebyshev_adaptive(CSR_matrix<double>& mat,
                                        std::vector<double>& x,
                                        std::vector<double>& b,
                                        sycl::queue& q,
                                        size_t degree = 4,
                                        size_t* used_degree = nullptr);

unsigned int CG_MKL_jacobi(CSR_matrix<double>& mat,
                           std::vector<double>& x,
                           std::vector<double>& b);

unsigned int CG_MKL_plain(CSR_matrix<double>& mat,
                          std::vector<double>& x,
                          std::vector<double>& b);

unsigned int CG_MKL_block_jacobi(CSR_matrix<double>& mat,
                                 std::vector<double>& x,
                                 std::vector<double>& b,
                                 size_t block_size = 16);

unsigned int CG_MKL_chebyshev(CSR_matrix<double>& mat,
                              std::vector<double>& x,
                              std::vector<double>& b,
                              size_t degree = 4,
                              size_t* used_degree = nullptr);

unsigned int CG_MKL_IC0(CSR_matrix<double>& mat,
                        std::vector<double>& x,
                        std::vector<double>& b,
                        const IC0Preconditioner& ic0);

void GMRES_SYCL(CSR_matrix<double>& mat,
            std::vector<double>& x, 
            std::vector<double>& b, 
            std::string& device); // не проверен

void CGS_SYCL(CSR_matrix<double>& mat,
            std::vector<double>& x, 
            std::vector<double>& b, 
            std::string& device); // не проверен

// -------------------- Math operations --------------------

double scalar_product(const std::vector<double>& a, 
                    const std::vector<double>& b);

void scalar_product_parallel(sycl::queue& q, 
                            sycl::buffer<double>& a_buf, 
                            sycl::buffer<double>& b_buf, 
                            sycl::buffer<double>& res_buf,
                            size_t n);

void CSR_mat_vec_prod_parallel(sycl::queue& q,
                            sycl::buffer<double>& vec_buf,
                            sycl::buffer<size_t>& row_ptr_buf,
                            sycl::buffer<size_t>& col_ind_buf,
                            sycl::buffer<double>& val_buf,
                            sycl::buffer<double>& res_buf,
                            size_t n);

// 32-bit-index overload — same semantics, halves col_ind bandwidth on GPU.
void CSR_mat_vec_prod_parallel(sycl::queue& q,
                            sycl::buffer<double>& vec_buf,
                            sycl::buffer<uint32_t>& row_ptr_buf,
                            sycl::buffer<uint32_t>& col_ind_buf,
                            sycl::buffer<double>& val_buf,
                            sycl::buffer<double>& res_buf,
                            size_t n);

// res = alpha * vec
void scale(sycl::queue&q,
            sycl::buffer<double>& vec_buf,
            sycl::buffer<double>& alpha_buf,
            sycl::buffer<double>& res_buf,
            size_t n);

// res = alpha * x + y
void axpy(sycl::queue& q,
            sycl::buffer<double>& x_buf,
            sycl::buffer<double>& y_buf,
            sycl::buffer<double>& alpha_buf,
            sycl::buffer<double>& res_buf,
            size_t n);

// res = x - y
void subtract(sycl::queue& q,
            sycl::buffer<double>& x_buf,
            sycl::buffer<double>& y_buf,
            sycl::buffer<double>& res_buf,
            size_t n);

void scale_division(sycl::queue& q,
                    sycl::buffer<double>& x_buf,
                    sycl::buffer<double>& alpha_buf,
                    sycl::buffer<double>& res_buf,
                    size_t n);
