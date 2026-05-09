#include "func.h"

#ifdef eps
constexpr double solver_eps = eps;
#undef eps
#else
constexpr double solver_eps = 1.e-13;
#endif

#include <mkl.h>

namespace {

struct MKLCsrMatrix {
    MKL_INT n = 0;
    std::vector<MKL_INT> row_ptr;
    std::vector<MKL_INT> col_ind;
    std::vector<double> values;

    explicit MKLCsrMatrix(CSR_matrix<double>& mat)
    {
        n = static_cast<MKL_INT>(mat.take_rows());

        const std::vector<size_t>& src_row_ptr = mat.row_ptr_ref();
        const std::vector<size_t>& src_col_ind = mat.col_ind_ref();
        const std::vector<double>& src_values = mat.val_ref();

        row_ptr.assign(static_cast<size_t>(n) + 1, 0);
        col_ind.clear();
        values.clear();
        col_ind.reserve(src_col_ind.size());
        values.reserve(src_values.size());

        for (MKL_INT row = 0; row < n; ++row) {
            std::vector<std::pair<size_t, double>> entries;
            entries.reserve(src_row_ptr[static_cast<size_t>(row) + 1] - src_row_ptr[static_cast<size_t>(row)]);
            for (size_t idx = src_row_ptr[static_cast<size_t>(row)];
                 idx < src_row_ptr[static_cast<size_t>(row) + 1];
                 ++idx) {
                entries.emplace_back(src_col_ind[idx], src_values[idx]);
            }

            std::stable_sort(entries.begin(), entries.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.first < rhs.first;
                });

            for (const auto& [col, value] : entries) {
                col_ind.push_back(static_cast<MKL_INT>(col + 1));
                values.push_back(value);
            }
            row_ptr[static_cast<size_t>(row) + 1] = static_cast<MKL_INT>(col_ind.size() + 1);
        }

        row_ptr[0] = 1;
    }
};

static void mkl_spmv(const MKLCsrMatrix& mat, const double* x, double* y)
{
    const char transa = 'N';
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    mkl_dcsrgemv(&transa, &mat.n, mat.values.data(), mat.row_ptr.data(), mat.col_ind.data(), x, y);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
}

static double vector_norm_sq(const std::vector<double>& vec)
{
    return scalar_product(vec, vec);
}

static void apply_diagonal_host(const std::vector<double>& inv_diag,
                                const double* r,
                                double* z)
{
    const size_t n = inv_diag.size();
    for (size_t i = 0; i < n; ++i) {
        z[i] = inv_diag[i] * r[i];
    }
}

static std::vector<double> build_jacobi_inverse(const std::vector<size_t>& row_ptr,
                                                const std::vector<size_t>& col_ind,
                                                const std::vector<double>& val)
{
    const size_t n = row_ptr.size() - 1;
    std::vector<double> inv_diag(n, 0.0);
    std::vector<size_t> diag_pos;
    build_diag_positions(row_ptr, col_ind, diag_pos);

    for (size_t i = 0; i < n; ++i) {
        const double diag = val[diag_pos[i]];
        inv_diag[i] = (std::abs(diag) > 1e-14) ? (1.0 / diag) : 0.0;
    }

    return inv_diag;
}

static double estimate_gershgorin_upper_bound_scaled(const std::vector<size_t>& row_ptr,
                                                     const std::vector<double>& val,
                                                     const std::vector<double>& inv_diag)
{
    const size_t n = inv_diag.size();
    return parallel_max_host<double>(0, n, 8192, 0.0, [&](size_t i) {
        double row_sum = 0.0;
        for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            row_sum += std::abs(val[k]);
        }
        return std::abs(inv_diag[i]) * row_sum;
    });
}

static double estimate_chebyshev_lambda_min_scaled_fast(const std::vector<size_t>& row_ptr,
                                                        const std::vector<size_t>& col_ind,
                                                        const std::vector<double>& val,
                                                        const std::vector<double>& inv_diag,
                                                        const std::vector<double>& rhs,
                                                        double lambda_max_estimate)
{
    if (lambda_max_estimate <= 1e-14) {
        return std::max(1e-14, 1e-6 * lambda_max_estimate);
    }

    std::vector<double> probe = rhs;
    double probe_norm_sq = vector_norm_sq(probe);
    if (probe_norm_sq < 1e-20) {
        probe.assign(rhs.size(), 1.0);
        probe_norm_sq = static_cast<double>(rhs.size());
    }

    std::vector<double> temp(rhs.size(), 0.0);
    const size_t n = probe.size();
    parallel_for_host(0, n, 16384, [&](size_t i) {
        double sum = 0.0;
        for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            sum += val[k] * probe[col_ind[k]];
        }
        temp[i] = sum;
    });

    const double rayleigh = parallel_sum_host<double>(0, n, 16384, 0.0, [&](size_t i) {
        return probe[i] * inv_diag[i] * temp[i];
    });

    const double lambda_min_estimate = (probe_norm_sq > 1e-30)
        ? (rayleigh / probe_norm_sq)
        : lambda_max_estimate;
    return std::clamp(lambda_min_estimate, 1e-14, 0.999999 * lambda_max_estimate);
}

static double estimate_scaled_lambda_max_power(const std::vector<size_t>& row_ptr,
                                               const std::vector<size_t>& col_ind,
                                               const std::vector<double>& val,
                                               const std::vector<double>& inv_diag,
                                               const std::vector<double>& rhs,
                                               size_t power_iterations = 12)
{
    const size_t n = inv_diag.size();
    if (n == 0) {
        return 0.0;
    }

    std::vector<double> x = rhs;
    double x_norm_sq = vector_norm_sq(x);
    if (x_norm_sq < 1e-20) {
        x.assign(n, 1.0);
        x_norm_sq = static_cast<double>(n);
    }

    const double x_norm = std::sqrt(std::max(x_norm_sq, 1e-300));
    parallel_for_host(0, n, 16384, [&](size_t i) {
        x[i] /= x_norm;
    });

    std::vector<double> sqrt_inv_diag(n, 0.0);
    std::vector<double> scaled_x(n, 0.0);
    std::vector<double> y(n, 0.0);

    parallel_for_host(0, n, 16384, [&](size_t i) {
        sqrt_inv_diag[i] = std::sqrt(std::max(inv_diag[i], 0.0));
    });

    double lambda_max_estimate = 0.0;
    for (size_t iter = 0; iter < power_iterations; ++iter) {
        parallel_for_host(0, n, 16384, [&](size_t i) {
            scaled_x[i] = sqrt_inv_diag[i] * x[i];
        });

        parallel_for_host(0, n, 16384, [&](size_t i) {
            double sum = 0.0;
            for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                sum += val[k] * scaled_x[col_ind[k]];
            }
            y[i] = sum;
        });

        parallel_for_host(0, n, 16384, [&](size_t i) {
            const double scaled_value = sqrt_inv_diag[i] * y[i];
            scaled_x[i] = scaled_value;
        });

        const double rayleigh = parallel_sum_host<double>(0, n, 16384, 0.0, [&](size_t i) {
            return x[i] * scaled_x[i];
        });
        const double y_norm_sq = parallel_sum_host<double>(0, n, 16384, 0.0, [&](size_t i) {
            return scaled_x[i] * scaled_x[i];
        });

        lambda_max_estimate = std::max(lambda_max_estimate, rayleigh);
        if (y_norm_sq < 1e-30) {
            break;
        }

        const double y_norm = std::sqrt(y_norm_sq);
        parallel_for_host(0, n, 16384, [&](size_t i) {
            x[i] = scaled_x[i] / y_norm;
        });
    }

    return lambda_max_estimate;
}

static void apply_block_jacobi_host(const std::vector<double>& block_inv,
                                    const double* r,
                                    double* z,
                                    size_t n,
                                    size_t block_size)
{
    parallel_for_host(0, n, 4096, [&](size_t i) {
        const size_t block_idx = i / block_size;
        const size_t block_start = block_idx * block_size;
        const size_t bs = std::min(block_size, n - block_start);

        double sum = 0.0;
        const size_t local_i = i - block_start;
        const size_t offset = block_idx * block_size * block_size + local_i * block_size;
        for (size_t j = 0; j < bs; ++j) {
            sum += block_inv[offset + j] * r[block_start + j];
        }
        z[i] = sum;
    });
}

static void apply_chebyshev_host(const MKLCsrMatrix& mat,
                                 const std::vector<double>& inv_diag,
                                 const double* r,
                                 double* z,
                                 std::vector<double>& t,
                                 std::vector<double>& w,
                                 std::vector<double>& tmp,
                                 const std::vector<double>& steps)
{
    const size_t n = t.size();
    if (steps.empty()) {
        parallel_for_host(0, n, 4096, [&](size_t i) {
            z[i] = inv_diag[i] * r[i];
        });
        return;
    }

    const double first_tau = steps.front();
    parallel_for_host(0, n, 4096, [&](size_t i) {
        t[i] = r[i];
        w[i] = inv_diag[i] * t[i];
        z[i] = first_tau * w[i];
    });

    for (size_t step = 1; step < steps.size(); ++step) {
        const double prev_tau = steps[step - 1];
        const double tau = steps[step];

        mkl_spmv(mat, w.data(), tmp.data());

        parallel_for_host(0, n, 4096, [&](size_t i) {
            t[i] -= prev_tau * tmp[i];
            w[i] = inv_diag[i] * t[i];
            z[i] += tau * w[i];
        });
    }
}

template <typename Fn>
static unsigned int run_with_mkl_max_threads(Fn&& fn)
{
    mkl_set_dynamic(0);
    const int max_threads = std::max(1, mkl_get_max_threads());
    const int previous_threads = mkl_set_num_threads_local(max_threads);

    try {
        const unsigned int result = fn();
        mkl_set_num_threads_local(previous_threads);
        return result;
    } catch (...) {
        mkl_set_num_threads_local(previous_threads);
        throw;
    }
}

template <typename PreconditionerFn>
static unsigned int run_mkl_cg(CSR_matrix<double>& mat,
                               std::vector<double>& x,
                               std::vector<double>& b,
                               PreconditionerFn&& apply_preconditioner)
{
    const MKL_INT n = static_cast<MKL_INT>(b.size());
    if (n == 0) {
        return 0;
    }

    const double b_norm_sq = vector_norm_sq(b);
    if (b_norm_sq < 1e-10) {
        std::fill(x.begin(), x.end(), 0.0);
        return 0;
    }

    MKLCsrMatrix mkl_mat(mat);
    std::vector<double> r = b;
    std::vector<double> z(static_cast<size_t>(n), 0.0);
    std::vector<double> p(static_cast<size_t>(n), 0.0);
    std::vector<double> Ap(static_cast<size_t>(n), 0.0);

    apply_preconditioner(r.data(), z.data());
    cblas_dcopy(n, z.data(), 1, p.data(), 1);

    double old_rr = cblas_ddot(n, r.data(), 1, z.data(), 1);
    if (!std::isfinite(old_rr) || std::abs(old_rr) < 1e-30) {
        throw std::runtime_error("MKL CG produced invalid initial residual");
    }

    unsigned int iteration = 0;
    const unsigned int max_iter = static_cast<unsigned int>(n);

    do {
        ++iteration;

        mkl_spmv(mkl_mat, p.data(), Ap.data());
        const double pAp = cblas_ddot(n, p.data(), 1, Ap.data(), 1);
        const double alpha = (std::abs(pAp) > 1e-30) ? old_rr / pAp : 0.0;

        cblas_daxpy(n, alpha, p.data(), 1, x.data(), 1);
        cblas_daxpy(n, -alpha, Ap.data(), 1, r.data(), 1);

        apply_preconditioner(r.data(), z.data());
        const double new_rr = cblas_ddot(n, r.data(), 1, z.data(), 1);
        if (!std::isfinite(new_rr)) {
            throw std::runtime_error("MKL CG produced invalid residual");
        }

        if (std::abs(new_rr) / b_norm_sq < solver_eps * solver_eps || iteration == max_iter) {
            break;
        }

        const double beta = (std::abs(old_rr) > 1e-30) ? new_rr / old_rr : 0.0;
        cblas_dscal(n, beta, p.data(), 1);
        cblas_daxpy(n, 1.0, z.data(), 1, p.data(), 1);
        old_rr = new_rr;
    } while (iteration < max_iter);

    return iteration;
}

template <typename PreconditionerFn>
static unsigned int run_mkl_native_dcg(CSR_matrix<double>& mat,
                                       std::vector<double>& x,
                                       std::vector<double>& b,
                                       bool use_preconditioner,
                                       PreconditionerFn&& apply_preconditioner)
{
    const MKL_INT n = static_cast<MKL_INT>(b.size());
    if (n == 0) {
        return 0;
    }

    const double b_norm_sq = vector_norm_sq(b);
    if (b_norm_sq < 1e-10) {
        std::fill(x.begin(), x.end(), 0.0);
        return 0;
    }

    MKLCsrMatrix mkl_mat(mat);
    MKL_INT rci_request = 0;
    MKL_INT ipar[128]{};
    double dpar[128]{};
    std::vector<double> tmp(static_cast<size_t>(4 * n), 0.0);

    dcg_init(&n, x.data(), b.data(), &rci_request, ipar, dpar, tmp.data());
    if (rci_request != 0) {
        throw std::runtime_error("dcg_init failed with request " + std::to_string(rci_request));
    }

    ipar[4] = std::max<MKL_INT>(150, n);
    ipar[7] = 1;
    ipar[8] = 1;
    ipar[9] = 0;
    ipar[10] = use_preconditioner ? 1 : 0;
    dpar[0] = solver_eps;
    dpar[1] = 0.0;

    dcg_check(&n, x.data(), b.data(), &rci_request, ipar, dpar, tmp.data());
    if (rci_request != 0) {
        throw std::runtime_error("dcg_check failed with request " + std::to_string(rci_request));
    }

    while (true) {
        dcg(&n, x.data(), b.data(), &rci_request, ipar, dpar, tmp.data());

        if (rci_request == 0) {
            break;
        }
        if (rci_request == 1) {
            mkl_spmv(mkl_mat, tmp.data(), tmp.data() + n);
            continue;
        }
        if (rci_request == 2) {
            continue;
        }
        if (rci_request == 3 && use_preconditioner) {
            apply_preconditioner(tmp.data() + 2 * n, tmp.data() + 3 * n);
            continue;
        }

        throw std::runtime_error("dcg returned unsupported request " + std::to_string(rci_request));
    }

    MKL_INT itercount = 0;
    dcg_get(&n, x.data(), b.data(), &rci_request, ipar, dpar, tmp.data(), &itercount);
    return static_cast<unsigned int>(itercount);
}

} // namespace

unsigned int CG_MKL_jacobi(CSR_matrix<double>& mat,
                           std::vector<double>& x,
                           std::vector<double>& b)
{
    const std::vector<size_t>& row_ptr = mat.row_ptr_ref();
    const std::vector<size_t>& col_ind = mat.col_ind_ref();
    const std::vector<double>& val = mat.val_ref();
    const std::vector<double> inv_diag = build_jacobi_inverse(row_ptr, col_ind, val);

    return run_with_mkl_max_threads([&]() {
        return run_mkl_cg(mat, x, b, [&](const double* r, double* z) {
            apply_diagonal_host(inv_diag, r, z);
        });
    });
}

unsigned int CG_MKL_plain(CSR_matrix<double>& mat,
                          std::vector<double>& x,
                          std::vector<double>& b)
{
    return run_with_mkl_max_threads([&]() {
        return run_mkl_cg(mat, x, b, [&](const double* r, double* z) {
            const size_t n = b.size();
            for (size_t i = 0; i < n; ++i) {
                z[i] = r[i];
            }
        });
    });
}

unsigned int CG_MKL_block_jacobi(CSR_matrix<double>& mat,
                                 std::vector<double>& x,
                                 std::vector<double>& b,
                                 size_t block_size)
{
    const std::vector<size_t>& row_ptr = mat.row_ptr_ref();
    const std::vector<size_t>& col_ind = mat.col_ind_ref();
    const std::vector<double>& val = mat.val_ref();
    std::vector<double> block_inv;
    compute_block_jacobi_preconditioner(row_ptr, col_ind, val, b.size(), block_size, block_inv);

    return run_with_mkl_max_threads([&]() {
        return run_mkl_cg(mat, x, b, [&](const double* r, double* z) {
            apply_block_jacobi_host(block_inv, r, z, b.size(), block_size);
        });
    });
}

unsigned int CG_MKL_chebyshev(CSR_matrix<double>& mat,
                              std::vector<double>& x,
                              std::vector<double>& b,
                              size_t degree,
                              size_t* used_degree)
{
    const std::vector<size_t>& row_ptr = mat.row_ptr_ref();
    const std::vector<size_t>& col_ind = mat.col_ind_ref();
    const std::vector<double>& val = mat.val_ref();
    const std::vector<double> inv_diag = build_jacobi_inverse(row_ptr, col_ind, val);
    const double gershgorin_lambda_max = estimate_gershgorin_upper_bound_scaled(row_ptr, val, inv_diag);

    if (gershgorin_lambda_max < 1e-14) {
        std::fill(x.begin(), x.end(), 0.0);
        return 0;
    }

    const double lambda_min_raw =
        estimate_chebyshev_lambda_min_scaled_fast(row_ptr, col_ind, val, inv_diag, b, gershgorin_lambda_max);
    const size_t n = b.size();
    const size_t power_iterations = (n >= 100000) ? 6 : ((n >= 50000) ? 8 : 12);
    const double power_safety = (n >= 100000) ? 1.35 : ((n >= 50000) ? 1.25 : 1.10);
    const double power_lambda_max =
        estimate_scaled_lambda_max_power(row_ptr, col_ind, val, inv_diag, b, power_iterations);
    const double lambda_max_bound = std::clamp(
        std::max(power_safety * power_lambda_max, 1.0001 * lambda_min_raw),
        1.0001 * lambda_min_raw,
        std::max(gershgorin_lambda_max, 1.0001 * lambda_min_raw));
    const double lambda_min_bound = lambda_min_raw;
    const size_t active_degree = std::max<size_t>(1, degree);
    if (used_degree != nullptr) {
        *used_degree = active_degree;
    }

    std::vector<double> chebyshev_steps;
    build_chebyshev_steps(lambda_min_bound, lambda_max_bound, active_degree, chebyshev_steps);

    MKLCsrMatrix mkl_mat(mat);
    std::vector<double> t(b.size(), 0.0);
    std::vector<double> w(b.size(), 0.0);
    std::vector<double> tmp(b.size(), 0.0);

    return run_with_mkl_max_threads([&]() {
        return run_mkl_cg(mat, x, b, [&](const double* r, double* z) {
            apply_chebyshev_host(mkl_mat, inv_diag, r, z, t, w, tmp, chebyshev_steps);
        });
    });
}

unsigned int CG_MKL_IC0(CSR_matrix<double>& mat,
                        std::vector<double>& x,
                        std::vector<double>& b,
                        const IC0Preconditioner& ic0)
{
    std::vector<double> rhs(b.size(), 0.0);
    std::vector<double> y(b.size(), 0.0);
    std::vector<double> z(b.size(), 0.0);

    return run_with_mkl_max_threads([&]() {
        return run_mkl_cg(mat, x, b, [&](const double* r, double* out) {
            for (size_t i = 0; i < b.size(); ++i) {
                rhs[i] = r[i];
            }
            applyIC0_preconditioner_host(ic0, rhs, y, z);
            for (size_t i = 0; i < b.size(); ++i) {
                out[i] = z[i];
            }
        });
    });
}
