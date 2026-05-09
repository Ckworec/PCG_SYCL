#include "func.h"
#include <memory>

static double scalar_product_host(const std::vector<double>& a,
                                  const std::vector<double>& b)
{
    const size_t n = a.size();
    return parallel_sum_host<double>(0, n, 32768, 0.0, [&](size_t i) {
        return a[i] * b[i];
    });
}

static void csr_mat_vec_prod_host(const std::vector<size_t>& row_ptr,
                                  const std::vector<size_t>& col_ind,
                                  const std::vector<double>& val,
                                  const std::vector<double>& x,
                                  std::vector<double>& y)
{
    const size_t n = y.size();
    parallel_for_host(0, n, 16384, [&](size_t i) {
        double sum = 0.0;
        for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            sum += val[k] * x[col_ind[k]];
        }
        y[i] = sum;
    });
}

static std::vector<double> build_jacobi_inverse_host_local(const std::vector<size_t>& row_ptr,
                                                           const std::vector<size_t>& diag_pos,
                                                           const std::vector<double>& val)
{
    const size_t n = diag_pos.size();
    std::vector<double> inv_diag(n, 0.0);

    parallel_for_host(0, n, 16384, [&](size_t i) {
        const double diag = val[diag_pos[i]];
        inv_diag[i] = (std::abs(diag) > 1e-14) ? (1.0 / diag) : 0.0;
    });

    return inv_diag;
}

static void initialize_diagonal_preconditioned_direction(sycl::queue& q,
                                                         sycl::buffer<double>& r_buf,
                                                         sycl::buffer<double>& M_inv_buf,
                                                         sycl::buffer<double>& z_buf,
                                                         sycl::buffer<double>& p_buf,
                                                         sycl::buffer<double>& rr_buf,
                                                         size_t n)
{
    q.submit([&](sycl::handler& h) {
        auto r = r_buf.get_access<sycl::access::mode::read>(h);
        auto M_inv = M_inv_buf.get_access<sycl::access::mode::read>(h);
        auto z = z_buf.get_access<sycl::access::mode::write>(h);
        auto p = p_buf.get_access<sycl::access::mode::write>(h);
        auto rr = sycl::reduction(rr_buf, h, sycl::plus<double>());

        h.parallel_for(sycl::range<1>(n), rr, [=](sycl::id<1> i, auto& sum) {
            const double zi = M_inv[i] * r[i];
            z[i] = zi;
            p[i] = zi;
            sum += r[i] * zi;
        });
    });
}

static void update_solution_residual_and_apply_diagonal(sycl::queue& q,
                                                        sycl::buffer<double>& x_buf,
                                                        sycl::buffer<double>& r_buf,
                                                        sycl::buffer<double>& z_buf,
                                                        sycl::buffer<double>& p_buf,
                                                        sycl::buffer<double>& Ap_buf,
                                                        sycl::buffer<double>& alpha_buf,
                                                        sycl::buffer<double>& M_inv_buf,
                                                        sycl::buffer<double>& rr_buf,
                                                        size_t n)
{
    q.submit([&](sycl::handler& h) {
        auto x = x_buf.get_access<sycl::access::mode::read_write>(h);
        auto r = r_buf.get_access<sycl::access::mode::read_write>(h);
        auto z = z_buf.get_access<sycl::access::mode::write>(h);
        auto p = p_buf.get_access<sycl::access::mode::read>(h);
        auto Ap = Ap_buf.get_access<sycl::access::mode::read>(h);
        auto alpha = alpha_buf.get_access<sycl::access::mode::read>(h);
        auto M_inv = M_inv_buf.get_access<sycl::access::mode::read>(h);
        auto rr = sycl::reduction(rr_buf, h, sycl::plus<double>());

        h.parallel_for(sycl::range<1>(n), rr, [=](sycl::id<1> i, auto& sum) {
            const double alpha_value = alpha[0];
            x[i] += alpha_value * p[i];
            const double residual = r[i] - alpha_value * Ap[i];
            r[i] = residual;
            const double zi = M_inv[i] * residual;
            z[i] = zi;
            sum += residual * zi;
        });
    });
}


unsigned int CG_SYCL_plain(CSR_matrix<double>& mat,
                           std::vector<double>& x,
                           std::vector<double>& b,
                           sycl::queue& q)
{
    double b_norm = scalar_product(b, b);
    if (b_norm < 1e-10) {
        std::fill(x.begin(), x.end(), 0.0);
        return 0;
    }

    auto& row_ptr = mat.row_ptr_ref();
    auto& col_ind = mat.col_ind_ref();
    auto& val = mat.val_ref();
    const size_t n = row_ptr.size() - 1;
    std::vector<double> r = b;
    std::vector<double> p(n, 0.0);
    std::vector<double> Ap(n, 0.0);
    double old_rr = 0.0;
    double new_rr = 0.0;
    double pAp = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    unsigned int iteration = 0;
    unsigned int max_iter = static_cast<unsigned int>(n);

    sycl::buffer<double> val_buf(val.data(), sycl::range<1>(val.size()));
    sycl::buffer<size_t> col_ind_buf(col_ind.data(), sycl::range<1>(col_ind.size()));
    sycl::buffer<size_t> row_ptr_buf(row_ptr.data(), sycl::range<1>(row_ptr.size()));
    sycl::buffer<double> r_buf(r.data(), sycl::range<1>(n));
    sycl::buffer<double> p_buf(p.data(), sycl::range<1>(n));
    sycl::buffer<double> x_buf(x.data(), sycl::range<1>(n));
    sycl::buffer<double> Ap_buf(Ap.data(), sycl::range<1>(n));
    sycl::buffer<double> old_rr_buf(&old_rr, sycl::range<1>(1));
    sycl::buffer<double> new_rr_buf(&new_rr, sycl::range<1>(1));
    sycl::buffer<double> pAp_buf(&pAp, sycl::range<1>(1));
    sycl::buffer<double> alpha_buf(&alpha, sycl::range<1>(1));
    sycl::buffer<double> beta_buf(&beta, sycl::range<1>(1));

    r_buf.set_final_data(nullptr);
    p_buf.set_final_data(nullptr);
    x_buf.set_final_data(nullptr);
    Ap_buf.set_final_data(nullptr);
    old_rr_buf.set_final_data(nullptr);
    new_rr_buf.set_final_data(nullptr);
    pAp_buf.set_final_data(nullptr);
    alpha_buf.set_final_data(nullptr);
    beta_buf.set_final_data(nullptr);

    q.submit([&](sycl::handler& h) {
        sycl::accessor r_access(r_buf, h, sycl::read_only);
        sycl::accessor p_access(p_buf, h, sycl::write_only, sycl::no_init);

        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            p_access[i] = r_access[i];
        });
    });

    scalar_product_parallel(q, r_buf, r_buf, old_rr_buf, n);

    do {
        iteration++;

        CSR_mat_vec_prod_parallel(q, p_buf, row_ptr_buf, col_ind_buf, val_buf, Ap_buf, n);
        scalar_product_parallel(q, p_buf, Ap_buf, pAp_buf, n);

        q.submit([&](sycl::handler& h) {
            sycl::accessor old_rr_access(old_rr_buf, h, sycl::read_only);
            sycl::accessor pAp_access(pAp_buf, h, sycl::read_write);
            sycl::accessor alpha_access(alpha_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor new_rr_access(new_rr_buf, h, sycl::write_only, sycl::no_init);

            h.single_task([=]() {
                alpha_access[0] = (std::abs(pAp_access[0]) > 1e-30) ? old_rr_access[0] / pAp_access[0] : 0.0;
                new_rr_access[0] = 0.0;
                pAp_access[0] = 0.0;
            });
        });

        q.submit([&](sycl::handler& h) {
            sycl::accessor x_access(x_buf, h, sycl::read_write);
            sycl::accessor r_access(r_buf, h, sycl::read_write);
            sycl::accessor p_access(p_buf, h, sycl::read_only);
            sycl::accessor Ap_access(Ap_buf, h, sycl::read_only);
            sycl::accessor alpha_access(alpha_buf, h, sycl::read_only);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                x_access[i] += alpha_access[0] * p_access[i];
                r_access[i] -= alpha_access[0] * Ap_access[i];
            });
        });

        scalar_product_parallel(q, r_buf, r_buf, new_rr_buf, n);

        q.submit([&](sycl::handler& h) {
            sycl::accessor acc(new_rr_buf, h, sycl::read_only);
            h.copy(acc, &new_rr);
        }).wait_and_throw();

        if (new_rr / b_norm < eps * eps || iteration == max_iter) {
            break;
        }

        q.submit([&](sycl::handler& h) {
            sycl::accessor old_rr_access(old_rr_buf, h, sycl::read_write);
            sycl::accessor new_rr_access(new_rr_buf, h, sycl::read_only);
            sycl::accessor beta_access(beta_buf, h, sycl::write_only, sycl::no_init);

            h.single_task([=]() {
                beta_access[0] = (std::abs(old_rr_access[0]) > 1e-30) ? new_rr_access[0] / old_rr_access[0] : 0.0;
                old_rr_access[0] = new_rr_access[0];
            });
        });

        q.submit([&](sycl::handler& h) {
            sycl::accessor p_access(p_buf, h, sycl::read_write);
            sycl::accessor r_access(r_buf, h, sycl::read_only);
            sycl::accessor beta_access(beta_buf, h, sycl::read_only);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                p_access[i] = r_access[i] + beta_access[0] * p_access[i];
            });
        });
    } while (new_rr / b_norm > eps * eps && iteration < max_iter);

    q.submit([&](sycl::handler& h) {
        sycl::accessor acc(x_buf, h, sycl::read_only);
        h.copy(acc, x.data());
    }).wait_and_throw();

    std::cout << "CG iterations: " << iteration << std::endl;
    return iteration;
}

unsigned int CG_SYCL_jacobi(CSR_matrix<double>& mat,
                            std::vector<double>& x,
                            std::vector<double>& b,
                            sycl::queue& q)
{
    double b_norm = scalar_product(b, b);

    if (b_norm < 1e-10) {
        std::fill(x.begin(), x.end(), 0.0);
        std::cout << "||b|| = 0" << std::endl;
        return 0;
    }

    auto& row_ptr = mat.row_ptr_ref();
    auto& col_ind = mat.col_ind_ref();
    auto& val = mat.val_ref();
    const size_t n = row_ptr.size() - 1;

    std::vector<double> r = b;
    std::vector<double> Ap(n);
    std::vector<double> z(n);
    std::vector<double> p(n);
    std::vector<double> M_inv(n, 0.0);
    std::vector<size_t> diag_pos;
    build_diag_positions(row_ptr, col_ind, diag_pos);

    unsigned int iteration = 0;
    unsigned int max_iter = static_cast<unsigned int>(row_ptr.size());

    double old_rr = 0.0;
    double new_rr = 0.0;
    double pAp = 0.0;
    double alpha = 0.0;
    double beta = 0.0;

    sycl::buffer<double> val_buf(val.data(), sycl::range<1>(val.size()));
    sycl::buffer<size_t> col_ind_buf(col_ind.data(), sycl::range<1>(col_ind.size()));
    sycl::buffer<size_t> row_ptr_buf(row_ptr.data(), sycl::range<1>(row_ptr.size()));
    sycl::buffer<size_t> diag_pos_buf(diag_pos.data(), sycl::range<1>(diag_pos.size()));
    sycl::buffer<double> r_buf(r.data(), sycl::range<1>(n));
    sycl::buffer<double> z_buf(z.data(), sycl::range<1>(n));
    sycl::buffer<double> M_inv_buf(M_inv.data(), sycl::range<1>(M_inv.size()));
    sycl::buffer<double> p_buf(p.data(), sycl::range<1>(n));
    sycl::buffer<double> x_buf(x.data(), sycl::range<1>(n));
    sycl::buffer<double> Ap_buf(Ap.data(), sycl::range<1>(n));

    sycl::buffer<double> old_rr_buf(&old_rr, sycl::range<1>(1));
    sycl::buffer<double> new_rr_buf(&new_rr, sycl::range<1>(1));
    sycl::buffer<double> pAp_buf(&pAp, sycl::range<1>(1));
    sycl::buffer<double> alpha_buf(&alpha, sycl::range<1>(1));
    sycl::buffer<double> beta_buf(&beta, sycl::range<1>(1));

    r_buf.set_final_data(nullptr);
    z_buf.set_final_data(nullptr);
    M_inv_buf.set_final_data(nullptr);
    p_buf.set_final_data(nullptr);
    x_buf.set_final_data(nullptr);
    Ap_buf.set_final_data(nullptr);
    old_rr_buf.set_final_data(nullptr);
    new_rr_buf.set_final_data(nullptr);
    pAp_buf.set_final_data(nullptr);
    alpha_buf.set_final_data(nullptr);
    beta_buf.set_final_data(nullptr);

    compute_jacobi_preconditioner_buf(row_ptr_buf, diag_pos_buf, val_buf, M_inv_buf, n, q);
    initialize_diagonal_preconditioned_direction(q, r_buf, M_inv_buf, z_buf, p_buf, old_rr_buf, n);

    do {
        iteration++;

        CSR_mat_vec_prod_parallel(q, p_buf, row_ptr_buf, col_ind_buf, val_buf, Ap_buf, n);
        scalar_product_parallel(q, p_buf, Ap_buf, pAp_buf, n);

        q.submit([&](sycl::handler& h) {
            sycl::accessor old_rr_access(old_rr_buf, h, sycl::read_only);
            sycl::accessor pAp_access(pAp_buf, h, sycl::read_write);
            sycl::accessor alpha_access(alpha_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor new_rr_access(new_rr_buf, h, sycl::write_only, sycl::no_init);

            h.single_task([=]() {
                alpha_access[0] = (std::abs(pAp_access[0]) > 1e-30) ? old_rr_access[0] / pAp_access[0] : 0.0;
                new_rr_access[0] = 0.0;
                pAp_access[0] = 0.0;
            });
        });

        update_solution_residual_and_apply_diagonal(
            q, x_buf, r_buf, z_buf, p_buf, Ap_buf, alpha_buf, M_inv_buf, new_rr_buf, n);

        q.submit([&](sycl::handler& h) {
            sycl::accessor acc(new_rr_buf, h, sycl::read_only);
            h.copy(acc, &new_rr);
        }).wait_and_throw();

        if (new_rr / b_norm < eps * eps || iteration == max_iter) {
            break;
        }

        q.submit([&](sycl::handler& h) {
            sycl::accessor old_rr_access(old_rr_buf, h, sycl::read_write);
            sycl::accessor new_rr_access(new_rr_buf, h, sycl::read_only);
            sycl::accessor beta_access(beta_buf, h, sycl::write_only, sycl::no_init);

            h.single_task([=]() {
                beta_access[0] = (std::abs(old_rr_access[0]) > 1e-30) ? new_rr_access[0] / old_rr_access[0] : 0.0;
                old_rr_access[0] = new_rr_access[0];
            });
        });

        q.submit([&](sycl::handler& h) {
            sycl::accessor p_access(p_buf, h, sycl::read_write);
            sycl::accessor z_access(z_buf, h, sycl::read_only);
            sycl::accessor beta_access(beta_buf, h, sycl::read_only);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                p_access[i] = z_access[i] + beta_access[0] * p_access[i];
            });
        });
    } while (new_rr / b_norm > eps * eps && iteration < max_iter);

    q.submit([&](sycl::handler& h) {
        sycl::accessor acc(x_buf, h, sycl::read_only);
        h.copy(acc, x.data());
    }).wait_and_throw();

    std::cout << "iterations: " << iteration << std::endl;
    return iteration;
}
// =====================================================================
// IC0-preconditioned CG -- port of CG_IC0_SYCL from ic0_sycl/ic0_cg_sycl.cpp.
//
// Two paths, selected by the SYCL device:
//   * GPU: every CG scalar lives on the device (alpha, beta, pAp, old_rr).
//     The Jacobi-sweep IC0 apply is invoked through
//     applyIC0_preconditioner_jacobi_device. Only new_rr crosses to the
//     host each iteration (it drives the convergence test); the final
//     solution vector is copied once at the end.
//   * CPU: bypass SYCL entirely. Submitting micro-kernels through the CPU
//     OpenCL queue is multiple ms per launch on this stack -- with ~9
//     kernels per CG iteration that turns into many seconds of overhead
//     for small/medium matrices. We run a plain host CG with the same
//     parallel helpers as the rest of the project, and apply uses
//     applyIC0_preconditioner_host (Jacobi by default; env IC0_JACOBI_SWEEPS
//     selects sweep count, 0 == exact level-scheduled triangular solve).
// =====================================================================

static unsigned int CG_IC0_host(const std::vector<size_t>& row_ptr,
                                const std::vector<size_t>& col_ind,
                                const std::vector<double>& val,
                                const IC0Preconditioner&   ic0,
                                std::vector<double>&       x,
                                const std::vector<double>& b)
{
    const double b_norm = scalar_product_host(b, b);
    if (b_norm < 1e-10) {
        std::fill(x.begin(), x.end(), 0.0);
        return 0;
    }

    const size_t n = b.size();
    std::vector<double> r = b;
    std::vector<double> y(n, 0.0);
    std::vector<double> z(n, 0.0);
    std::vector<double> p(n, 0.0);
    std::vector<double> Ap(n, 0.0);

    applyIC0_preconditioner_host(ic0, r, y, z);
    double old_rr = scalar_product_host(r, z);
    if (!std::isfinite(old_rr) || old_rr < 0.0) {
        throw std::runtime_error("IC0 produced invalid initial residual: "
                                 + std::to_string(old_rr));
    }
    p = z;

    unsigned int iteration = 0;
    const unsigned int max_iter = static_cast<unsigned int>(n);
    do {
        ++iteration;
        csr_mat_vec_prod_host(row_ptr, col_ind, val, p, Ap);
        const double pAp   = scalar_product_host(p, Ap);
        const double alpha = (std::abs(pAp) > 1e-30) ? old_rr / pAp : 0.0;

        parallel_for_host(0, n, 16384, [&](size_t i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        });

        applyIC0_preconditioner_host(ic0, r, y, z);
        const double new_rr = scalar_product_host(r, z);
        if (!std::isfinite(new_rr)) {
            throw std::runtime_error("IC0 produced invalid residual during iteration: "
                                     + std::to_string(new_rr));
        }
        if (new_rr / b_norm < eps * eps || iteration == max_iter) break;

        const double beta = (std::abs(old_rr) > 1e-30) ? new_rr / old_rr : 0.0;
        parallel_for_host(0, n, 16384, [&](size_t i) {
            p[i] = z[i] + beta * p[i];
        });
        old_rr = new_rr;
    } while (iteration < max_iter);

    std::cout << "CG-IC iterations: " << iteration << std::endl;
    return iteration;
}

unsigned int CG_SYCL_IC0(CSR_matrix<double>& mat,
                         std::vector<double>& x,
                         std::vector<double>& b,
                         sycl::queue& q)
{
    auto& row_ptr = mat.row_ptr_ref();
    auto& col_ind = mat.col_ind_ref();
    auto& val = mat.val_ref();
    const size_t n = row_ptr.size() - 1;
    IC0Preconditioner ic0;
    ic0_factor(n, row_ptr, col_ind, val, ic0);
    return CG_SYCL_IC0(mat, x, b, q, ic0);
}

unsigned int CG_SYCL_IC0(CSR_matrix<double>& mat,
                         std::vector<double>& x,
                         std::vector<double>& b,
                         sycl::queue& q,
                         const IC0Preconditioner& ic0)
{
    auto& row_ptr = mat.row_ptr_ref();
    auto& col_ind = mat.col_ind_ref();
    auto& val     = mat.val_ref();
    const size_t n = row_ptr.size() - 1;

    if (q.get_device().is_cpu()) {
        return CG_IC0_host(row_ptr, col_ind, val, ic0, x, b);
    }

    const double b_norm = scalar_product(b, b);
    if (b_norm < 1e-10) {
        std::fill(x.begin(), x.end(), 0.0);
        return 0;
    }

    // Sanity-check the factor: NaN/Inf in L would silently corrupt the
    // solve and is much easier to diagnose here.
    for (size_t idx = 0; idx < ic0.L_vals.size(); ++idx) {
        if (!std::isfinite(ic0.L_vals[idx])) {
            size_t bad_row = 0;
            while (bad_row + 1 < row_ptr.size() && row_ptr[bad_row + 1] <= idx) {
                ++bad_row;
            }
            throw std::runtime_error("IC0 factor contains NaN/Inf at row "
                                     + std::to_string(bad_row)
                                     + ", col " + std::to_string(col_ind[idx]));
        }
    }

    int jacobi_sweeps = 2;
    if (const char* env = std::getenv("IC0_JACOBI_SWEEPS")) {
        const int v = std::atoi(env);
        if (v >= 1 && v <= 10) jacobi_sweeps = v;
        else if (v == 0)       jacobi_sweeps = 1; // exact path not implemented on GPU
    }

    std::vector<double> r = b;
    std::vector<double> Ap(n, 0.0);
    std::vector<double> z(n, 0.0);
    std::vector<double> p(n, 0.0);
    std::vector<double> y_temp(n, 0.0);
    std::vector<double> y2_temp(n, 0.0);

    sycl::buffer<double> r_buf {r.data(),       sycl::range<1>(n)};
    sycl::buffer<double> Ap_buf{Ap.data(),      sycl::range<1>(n)};
    sycl::buffer<double> z_buf {z.data(),       sycl::range<1>(n)};
    sycl::buffer<double> p_buf {p.data(),       sycl::range<1>(n)};
    sycl::buffer<double> x_buf {x.data(),       sycl::range<1>(n)};
    sycl::buffer<double> y_buf {y_temp.data(),  sycl::range<1>(n)};
    sycl::buffer<double> y2_buf{y2_temp.data(), sycl::range<1>(n)};

    // SpMV uses 32-bit indices (halves col_ind bandwidth on the GPU); the
    // IC0 apply itself stays on size_t because its tables hold size_t.
    auto& row_ptr_u32 = mat.row_ptr_u32_ref();
    auto& col_ind_u32 = mat.col_ind_u32_ref();
    sycl::buffer<uint32_t> row_ptr32_buf{row_ptr_u32.data(), sycl::range<1>(row_ptr_u32.size())};
    sycl::buffer<uint32_t> col_ind32_buf{col_ind_u32.data(), sycl::range<1>(col_ind_u32.size())};
    sycl::buffer<double>   val_buf      {val.data(),         sycl::range<1>(val.size())};

    sycl::buffer<size_t> ic0_row_ptr_buf      {ic0.row_ptr.data(),       sycl::range<1>(ic0.row_ptr.size())};
    sycl::buffer<size_t> ic0_col_idx_buf      {ic0.col_idx.data(),       sycl::range<1>(ic0.col_idx.size())};
    sycl::buffer<double> ic0_L_buf            {ic0.L_vals.data(),        sycl::range<1>(ic0.L_vals.size())};
    sycl::buffer<double> ic0_diag_buf         {ic0.diag.data(),          sycl::range<1>(ic0.diag.size())};
    sycl::buffer<size_t> ic0_diag_pos_buf     {ic0.diag_pos.data(),      sycl::range<1>(ic0.diag_pos.size())};
    sycl::buffer<size_t> ic0_upper_ptr_buf    {ic0.upper_ptr.data(),     sycl::range<1>(ic0.upper_ptr.size())};
    sycl::buffer<size_t> ic0_upper_row_idx_buf{ic0.upper_row_idx.data(), sycl::range<1>(ic0.upper_row_idx.size())};
    sycl::buffer<size_t> ic0_upper_pos_buf    {ic0.upper_pos.data(),     sycl::range<1>(ic0.upper_pos.size())};

    // All scalar state lives on the device. Only `new_rr` is fetched to
    // the host every iteration to drive the convergence test.
    double new_rr = 0.0;
    sycl::buffer<double> old_rr_buf{sycl::range<1>(1)};
    sycl::buffer<double> new_rr_buf{sycl::range<1>(1)};
    sycl::buffer<double> pAp_buf   {sycl::range<1>(1)};
    sycl::buffer<double> alpha_buf {sycl::range<1>(1)};
    sycl::buffer<double> beta_buf  {sycl::range<1>(1)};

    r_buf .set_final_data(nullptr);
    Ap_buf.set_final_data(nullptr);
    z_buf .set_final_data(nullptr);
    p_buf .set_final_data(nullptr);
    y_buf .set_final_data(nullptr);
    y2_buf.set_final_data(nullptr);
    old_rr_buf.set_final_data(nullptr);
    new_rr_buf.set_final_data(nullptr);
    pAp_buf   .set_final_data(nullptr);
    alpha_buf .set_final_data(nullptr);
    beta_buf  .set_final_data(nullptr);

    auto zero_scalar = [&](sycl::buffer<double>& buf) {
        q.submit([&](sycl::handler& h) {
            sycl::accessor acc(buf, h, sycl::write_only, sycl::no_init);
            h.single_task([=]() { acc[0] = 0.0; });
        });
    };

    auto apply_ic0 = [&](sycl::buffer<double>& src, sycl::buffer<double>& dst) {
        applyIC0_preconditioner_jacobi_device(q,
                                              ic0_row_ptr_buf, ic0_col_idx_buf,
                                              ic0_L_buf, ic0_diag_buf, ic0_diag_pos_buf,
                                              ic0_upper_ptr_buf,
                                              ic0_upper_row_idx_buf,
                                              ic0_upper_pos_buf,
                                              src, y_buf, y2_buf, dst,
                                              n, jacobi_sweeps);
    };

    apply_ic0(r_buf, z_buf);

    // p = z
    q.submit([&](sycl::handler& h) {
        sycl::accessor z_acc(z_buf, h, sycl::read_only);
        sycl::accessor p_acc(p_buf, h, sycl::write_only, sycl::no_init);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { p_acc[i] = z_acc[i]; });
    });

    zero_scalar(old_rr_buf);
    scalar_product_parallel(q, r_buf, z_buf, old_rr_buf, n);

    unsigned int       iteration = 0;
    const unsigned int max_iter  = static_cast<unsigned int>(n);

    do {
        ++iteration;

        CSR_mat_vec_prod_parallel(q, p_buf, row_ptr32_buf, col_ind32_buf, val_buf, Ap_buf, n);
        zero_scalar(pAp_buf);
        scalar_product_parallel(q, p_buf, Ap_buf, pAp_buf, n);

        // alpha = old_rr / pAp on the device.
        q.submit([&](sycl::handler& h) {
            sycl::accessor old_rr_acc(old_rr_buf, h, sycl::read_only);
            sycl::accessor pAp_acc   (pAp_buf,    h, sycl::read_only);
            sycl::accessor alpha_acc (alpha_buf,  h, sycl::write_only, sycl::no_init);
            h.single_task([=]() {
                const double pv = pAp_acc[0];
                alpha_acc[0] = (std::abs(pv) > 1e-30) ? old_rr_acc[0] / pv : 0.0;
            });
        });

        // x += alpha p ; r -= alpha Ap
        q.submit([&](sycl::handler& h) {
            sycl::accessor x_acc    (x_buf,     h, sycl::read_write);
            sycl::accessor r_acc    (r_buf,     h, sycl::read_write);
            sycl::accessor p_acc    (p_buf,     h, sycl::read_only);
            sycl::accessor Ap_acc   (Ap_buf,    h, sycl::read_only);
            sycl::accessor alpha_acc(alpha_buf, h, sycl::read_only);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                const double a = alpha_acc[0];
                x_acc[i] += a * p_acc[i];
                r_acc[i] -= a * Ap_acc[i];
            });
        });

        apply_ic0(r_buf, z_buf);
        zero_scalar(new_rr_buf);
        scalar_product_parallel(q, r_buf, z_buf, new_rr_buf, n);

        // Pull new_rr to the host -- the only h2d sync per iteration.
        q.submit([&](sycl::handler& h) {
            sycl::accessor acc(new_rr_buf, h, sycl::read_only);
            h.copy(acc, &new_rr);
        }).wait_and_throw();

        if (!std::isfinite(new_rr)) {
            throw std::runtime_error("IC0 produced invalid residual during iteration: "
                                     + std::to_string(new_rr));
        }
        if (new_rr / b_norm < eps * eps || iteration == max_iter) break;

        // beta = new_rr / old_rr ; old_rr <- new_rr  (all on device).
        q.submit([&](sycl::handler& h) {
            sycl::accessor old_rr_acc(old_rr_buf, h, sycl::read_write);
            sycl::accessor new_rr_acc(new_rr_buf, h, sycl::read_only);
            sycl::accessor beta_acc  (beta_buf,   h, sycl::write_only, sycl::no_init);
            h.single_task([=]() {
                const double old_v = old_rr_acc[0];
                beta_acc[0]   = (std::abs(old_v) > 1e-30) ? new_rr_acc[0] / old_v : 0.0;
                old_rr_acc[0] = new_rr_acc[0];
            });
        });

        // p = z + beta p
        q.submit([&](sycl::handler& h) {
            sycl::accessor p_acc   (p_buf,    h, sycl::read_write);
            sycl::accessor z_acc   (z_buf,    h, sycl::read_only);
            sycl::accessor beta_acc(beta_buf, h, sycl::read_only);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                p_acc[i] = z_acc[i] + beta_acc[0] * p_acc[i];
            });
        });
    } while (iteration < max_iter);

    // Final solution: the only x copy back to the host.
    q.submit([&](sycl::handler& h) {
        sycl::accessor acc(x_buf, h, sycl::read_only);
        h.copy(acc, x.data());
    }).wait_and_throw();

    std::cout << "CG-IC iterations: " << iteration << std::endl;
    return iteration;
}




unsigned int CG_SYCL_block_jacobi(CSR_matrix<double>& mat,
                                  std::vector<double>& x,
                                  std::vector<double>& b,
                                  sycl::queue& q,
                                  size_t block_size)
{
    double b_norm = scalar_product(b, b);
    if (b_norm < 1e-10) {
        std::fill(x.begin(), x.end(), 0.0);
        return 0;
    }

    auto& row_ptr = mat.row_ptr_ref();
    auto& col_ind = mat.col_ind_ref();
    auto& val = mat.val_ref();
    const size_t n = row_ptr.size() - 1;
    std::vector<double> r = b;
    std::vector<double> Ap(n), z(n), p(n);

    const size_t num_blocks = (n + block_size - 1) / block_size;
    std::vector<double> block_inv(num_blocks * block_size * block_size, 0.0);

    sycl::buffer<double> r_buf{r.data(), sycl::range<1>(n)};
    sycl::buffer<double> z_buf{z.data(), sycl::range<1>(n)};
    sycl::buffer<double> p_buf{p.data(), sycl::range<1>(n)};
    sycl::buffer<double> x_buf{x.data(), sycl::range<1>(n)};
    sycl::buffer<double> Ap_buf{Ap.data(), sycl::range<1>(n)};
    sycl::buffer<size_t> row_ptr_buf{row_ptr.data(), sycl::range<1>(row_ptr.size())};
    sycl::buffer<size_t> col_ind_buf{col_ind.data(), sycl::range<1>(col_ind.size())};
    sycl::buffer<double> val_buf{val.data(), sycl::range<1>(val.size())};
    sycl::buffer<double> block_inv_buf{block_inv.data(), sycl::range<1>(block_inv.size())};

    double old_rr = 0.0, new_rr = 0.0, pAp = 0.0, alpha = 0.0, beta = 0.0;
    sycl::buffer<double> old_rr_buf{&old_rr, sycl::range<1>(1)};
    sycl::buffer<double> new_rr_buf{&new_rr, sycl::range<1>(1)};
    sycl::buffer<double> pAp_buf{&pAp, sycl::range<1>(1)};
    sycl::buffer<double> alpha_buf{&alpha, sycl::range<1>(1)};
    sycl::buffer<double> beta_buf{&beta, sycl::range<1>(1)};

    r_buf.set_final_data(nullptr);
    z_buf.set_final_data(nullptr);
    p_buf.set_final_data(nullptr);
    x_buf.set_final_data(nullptr);
    Ap_buf.set_final_data(nullptr);
    block_inv_buf.set_final_data(nullptr);
    old_rr_buf.set_final_data(nullptr);
    new_rr_buf.set_final_data(nullptr);
    pAp_buf.set_final_data(nullptr);
    alpha_buf.set_final_data(nullptr);
    beta_buf.set_final_data(nullptr);

    compute_block_jacobi_preconditioner_device(q,
                                               row_ptr_buf,
                                               col_ind_buf,
                                               val_buf,
                                               block_inv_buf,
                                               n,
                                               block_size);

    apply_block_jacobi_preconditioner(q, block_inv_buf, r_buf, z_buf, n, block_size);

    q.submit([&](sycl::handler& h) {
        auto z_acc = z_buf.get_access<sycl::access::mode::read>(h);
        auto p_acc = p_buf.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { p_acc[i] = z_acc[i]; });
    });

    scalar_product_parallel(q, r_buf, z_buf, old_rr_buf, n);

    unsigned int iteration = 0;
    unsigned int max_iter = static_cast<unsigned int>(n);

    do {
        iteration++;

        CSR_mat_vec_prod_parallel(q, p_buf, row_ptr_buf, col_ind_buf, val_buf, Ap_buf, n);
        scalar_product_parallel(q, p_buf, Ap_buf, pAp_buf, n);

        q.submit([&](sycl::handler& h) {
            auto old_rr_acc = old_rr_buf.get_access<sycl::access::mode::read>(h);
            auto pAp_acc = pAp_buf.get_access<sycl::access::mode::read_write>(h);
            auto alpha_acc = alpha_buf.get_access<sycl::access::mode::write>(h);
            auto new_rr_acc = new_rr_buf.get_access<sycl::access::mode::write>(h);
            h.single_task([=]() {
                alpha_acc[0] = (std::abs(pAp_acc[0]) > 1e-30) ? old_rr_acc[0] / pAp_acc[0] : 0.0;
                new_rr_acc[0] = 0.0;
                pAp_acc[0] = 0.0;
            });
        });

        q.submit([&](sycl::handler& h) {
            auto x_acc = x_buf.get_access<sycl::access::mode::read_write>(h);
            auto r_acc = r_buf.get_access<sycl::access::mode::read_write>(h);
            auto p_acc = p_buf.get_access<sycl::access::mode::read>(h);
            auto Ap_acc = Ap_buf.get_access<sycl::access::mode::read>(h);
            auto alpha_acc = alpha_buf.get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                x_acc[i] += alpha_acc[0] * p_acc[i];
                r_acc[i] -= alpha_acc[0] * Ap_acc[i];
            });
        });

        apply_block_jacobi_preconditioner(q, block_inv_buf, r_buf, z_buf, n, block_size);
        scalar_product_parallel(q, r_buf, z_buf, new_rr_buf, n);

        q.submit([&](sycl::handler& h) {
            auto acc = new_rr_buf.get_access<sycl::access::mode::read>(h);
            h.copy(acc, &new_rr);
        }).wait_and_throw();

        if (new_rr / b_norm < eps * eps || iteration == max_iter) {
            break;
        }

        q.submit([&](sycl::handler& h) {
            auto old_rr_acc = old_rr_buf.get_access<sycl::access::mode::read_write>(h);
            auto new_rr_acc = new_rr_buf.get_access<sycl::access::mode::read>(h);
            auto beta_acc = beta_buf.get_access<sycl::access::mode::write>(h);
            h.single_task([=]() {
                beta_acc[0] = (std::abs(old_rr_acc[0]) > 1e-30) ? new_rr_acc[0] / old_rr_acc[0] : 0.0;
                old_rr_acc[0] = new_rr_acc[0];
            });
        });

        q.submit([&](sycl::handler& h) {
            auto p_acc = p_buf.get_access<sycl::access::mode::read_write>(h);
            auto z_acc = z_buf.get_access<sycl::access::mode::read>(h);
            auto beta_acc = beta_buf.get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                p_acc[i] = z_acc[i] + beta_acc[0] * p_acc[i];
            });
        });

    } while (new_rr / b_norm > eps * eps && iteration < max_iter);

    q.submit([&](sycl::handler& h) {
        auto acc = x_buf.get_access<sycl::access::mode::read>(h);
        h.copy(acc, x.data());
    }).wait_and_throw();
    std::cout << "CG-BlockJacobi iterations: " << iteration << std::endl;
    return iteration;
}


// ---------------- Adaptive Chebyshev (Lanczos-derived bounds) -------------------

// In-house extreme-eigenvalue solver for symmetric tridiagonal T (diag d, off-diag e).
// Power iteration for lambda_max; shifted power iteration for lambda_min.
// The matrices we care about have k <= a few hundred, so this is microseconds on host.
static void compute_tridiag_extreme_eigenvalues(const std::vector<double>& d,
                                                const std::vector<double>& e,
                                                double* out_lambda_min,
                                                double* out_lambda_max)
{
    const size_t k = d.size();
    if (k == 0) {
        *out_lambda_min = 1.0;
        *out_lambda_max = 1.0;
        return;
    }
    if (k == 1) {
        *out_lambda_min = d[0];
        *out_lambda_max = d[0];
        return;
    }

    // Gershgorin disk to bracket spectrum
    double g_min = std::numeric_limits<double>::infinity();
    double g_max = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < k; ++i) {
        double r = 0.0;
        if (i > 0)         r += std::abs(e[i - 1]);
        if (i + 1 < k)     r += std::abs(e[i]);
        g_min = std::min(g_min, d[i] - r);
        g_max = std::max(g_max, d[i] + r);
    }

    auto matvec_T = [&](const std::vector<double>& v, std::vector<double>& w) {
        for (size_t i = 0; i < k; ++i) {
            double s = d[i] * v[i];
            if (i > 0)     s += e[i - 1] * v[i - 1];
            if (i + 1 < k) s += e[i] * v[i + 1];
            w[i] = s;
        }
    };

    auto power_max = [&]() {
        std::vector<double> v(k), w(k);
        for (size_t i = 0; i < k; ++i) v[i] = 1.0 / std::sqrt(static_cast<double>(k));
        double lambda = g_max;
        for (int it = 0; it < 200; ++it) {
            matvec_T(v, w);
            double norm = 0.0;
            for (double x : w) norm += x * x;
            norm = std::sqrt(norm);
            if (norm < 1e-30) break;
            double lambda_new = 0.0;
            for (size_t i = 0; i < k; ++i) lambda_new += v[i] * w[i];
            const double inv_norm = 1.0 / norm;
            for (size_t i = 0; i < k; ++i) v[i] = w[i] * inv_norm;
            if (std::abs(lambda_new - lambda) < 1e-10 * std::abs(lambda_new) + 1e-14) {
                lambda = lambda_new;
                break;
            }
            lambda = lambda_new;
        }
        return lambda;
    };

    // shift > lambda_max; eigenvalues of (shift*I - T) are (shift - lambda_i),
    // largest = shift - lambda_min => lambda_min = shift - largest.
    auto power_min_shifted = [&](double shift) {
        std::vector<double> v(k), w(k);
        // Non-uniform init to avoid landing exactly on an unhelpful invariant subspace
        double sqsum = 0.0;
        for (size_t i = 0; i < k; ++i) {
            v[i] = std::sin(0.13 * (static_cast<double>(i) + 1.0));
            sqsum += v[i] * v[i];
        }
        const double n0 = (sqsum > 0.0) ? 1.0 / std::sqrt(sqsum) : 1.0;
        for (auto& x : v) x *= n0;

        double lambda = shift - g_min;
        for (int it = 0; it < 200; ++it) {
            matvec_T(v, w);
            for (size_t i = 0; i < k; ++i) w[i] = shift * v[i] - w[i];
            double norm = 0.0;
            for (double x : w) norm += x * x;
            norm = std::sqrt(norm);
            if (norm < 1e-30) break;
            double lambda_new = 0.0;
            for (size_t i = 0; i < k; ++i) lambda_new += v[i] * w[i];
            const double inv_norm = 1.0 / norm;
            for (size_t i = 0; i < k; ++i) v[i] = w[i] * inv_norm;
            if (std::abs(lambda_new - lambda) < 1e-10 * std::abs(lambda_new) + 1e-14) {
                lambda = lambda_new;
                break;
            }
            lambda = lambda_new;
        }
        return shift - lambda;
    };

    double lambda_max = power_max();
    const double shift = std::max(2.0 * lambda_max, 2.0 * g_max) + 1.0;
    double lambda_min = power_min_shifted(shift);

    if (lambda_min > lambda_max) std::swap(lambda_min, lambda_max);
    lambda_min = std::clamp(lambda_min, g_min, g_max);
    lambda_max = std::clamp(lambda_max, g_min, g_max);

    *out_lambda_min = lambda_min;
    *out_lambda_max = lambda_max;
}

unsigned int CG_SYCL_chebyshev_adaptive(CSR_matrix<double>& mat,
                                        std::vector<double>& x,
                                        std::vector<double>& b,
                                        sycl::queue& q,
                                        size_t degree,
                                        size_t* used_degree)
{
    // Strategy: run pure Jacobi-PCG for WARMUP_ITERS to collect the Lanczos
    // tridiagonal of D^{-1}A; then derive lambda_min/max from its extreme
    // eigenvalues and switch to Chebyshev preconditioning with those bounds.
    // No expensive power-iteration setup is performed.
    double b_norm = scalar_product(b, b);
    if (b_norm < 1e-10) {
        if (used_degree != nullptr) {
            *used_degree = 0;
        }
        std::fill(x.begin(), x.end(), 0.0);
        return 0;
    }

    auto& row_ptr = mat.row_ptr_ref();
    auto& col_ind = mat.col_ind_ref();
    auto& val = mat.val_ref();
    const size_t n = row_ptr.size() - 1;
    std::vector<double> r = b;
    std::vector<double> Ap(n), z(n), p(n);

    auto setup_start = std::chrono::high_resolution_clock::now();

    const bool on_gpu = q.get_device().is_gpu();

    std::vector<double> M_inv_host;
    std::unique_ptr<sycl::buffer<double>> M_inv_buf_ptr;
    double lambda_min_bound = 0.0;
    double lambda_max_bound = 0.0;

    std::unique_ptr<sycl::buffer<uint32_t>> row_ptr_u32_buf_ptr;
    std::unique_ptr<sycl::buffer<uint32_t>> col_ind_u32_buf_ptr;
    std::unique_ptr<sycl::buffer<size_t>> row_ptr_buf_ptr;
    std::unique_ptr<sycl::buffer<size_t>> col_ind_buf_ptr;
    sycl::buffer<double> val_buf{val.data(), sycl::range<1>(val.size())};

    if (on_gpu) {
        auto& row_ptr_u32 = mat.row_ptr_u32_ref();
        auto& col_ind_u32 = mat.col_ind_u32_ref();
        row_ptr_u32_buf_ptr = std::make_unique<sycl::buffer<uint32_t>>(row_ptr_u32.data(), sycl::range<1>(row_ptr_u32.size()));
        col_ind_u32_buf_ptr = std::make_unique<sycl::buffer<uint32_t>>(col_ind_u32.data(), sycl::range<1>(col_ind_u32.size()));
        row_ptr_u32_buf_ptr->set_final_data(nullptr);
        col_ind_u32_buf_ptr->set_final_data(nullptr);

        // Build M_inv = 1/diag(A) on device with a tiny fused kernel — no power iteration.
        M_inv_buf_ptr = std::make_unique<sycl::buffer<double>>(sycl::range<1>(n));
        M_inv_buf_ptr->set_final_data(nullptr);
        q.submit([&](sycl::handler& h) {
            sycl::accessor row_ptr_acc(*row_ptr_u32_buf_ptr, h, sycl::read_only);
            sycl::accessor col_ind_acc(*col_ind_u32_buf_ptr, h, sycl::read_only);
            sycl::accessor val_acc(val_buf, h, sycl::read_only);
            sycl::accessor minv_acc(*M_inv_buf_ptr, h, sycl::write_only, sycl::no_init);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                const uint32_t row = static_cast<uint32_t>(i[0]);
                const uint32_t begin = row_ptr_acc[row];
                const uint32_t end = row_ptr_acc[row + 1];
                double diag = 0.0;
                for (uint32_t k = begin; k < end; ++k) {
                    if (col_ind_acc[k] == row) { diag = val_acc[k]; break; }
                }
                minv_acc[row] = (sycl::fabs(diag) > 1e-14) ? (1.0 / diag) : 0.0;
            });
        });
    }
    else {
        row_ptr_buf_ptr = std::make_unique<sycl::buffer<size_t>>(row_ptr.data(), sycl::range<1>(row_ptr.size()));
        col_ind_buf_ptr = std::make_unique<sycl::buffer<size_t>>(col_ind.data(), sycl::range<1>(col_ind.size()));
        std::vector<size_t> diag_pos;
        build_diag_positions(row_ptr, col_ind, diag_pos);
        M_inv_host = build_jacobi_inverse_host_local(row_ptr, diag_pos, val);
        M_inv_buf_ptr = std::make_unique<sycl::buffer<double>>(M_inv_host.data(), sycl::range<1>(n));
        M_inv_buf_ptr->set_final_data(nullptr);
    }

    const size_t active_degree = std::max<size_t>(1, degree);
    if (used_degree != nullptr) {
        *used_degree = active_degree;
    }

    // Steps are empty until warmup completes -> Jacobi only during warmup.
    std::vector<double> chebyshev_steps;
    double first_tau_host = 1.0;

    {
        auto setup_end = std::chrono::high_resolution_clock::now();
        const double setup_ms = std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
        std::cout << "Chebyshev-adaptive setup [" << (on_gpu ? "GPU" : "CPU") << "]: " << setup_ms
                  << " ms (no power iteration; bounds from Lanczos warmup)" << std::endl;
    }

    sycl::buffer<double> r_buf{r.data(), sycl::range<1>(n)};
    sycl::buffer<double> z_buf{z.data(), sycl::range<1>(n)};
    sycl::buffer<double> p_buf{p.data(), sycl::range<1>(n)};
    sycl::buffer<double> x_buf{x.data(), sycl::range<1>(n)};
    sycl::buffer<double> Ap_buf{Ap.data(), sycl::range<1>(n)};
    std::vector<double> t(n, 0.0);
    sycl::buffer<double> t_buf{t.data(), sycl::range<1>(n)};
    sycl::buffer<double>& M_inv_buf = *M_inv_buf_ptr;

    sycl::buffer<double> first_tau_buf{&first_tau_host, sycl::range<1>(1)};

    r_buf.set_final_data(nullptr);
    z_buf.set_final_data(nullptr);
    p_buf.set_final_data(nullptr);
    x_buf.set_final_data(nullptr);
    Ap_buf.set_final_data(nullptr);
    t_buf.set_final_data(nullptr);
    M_inv_buf.set_final_data(nullptr);
    first_tau_buf.set_final_data(nullptr);

    std::cout << "Chebyshev-adaptive degree: " << active_degree << std::endl;

    auto spmv_p_to_Ap = [&]() {
        if (on_gpu) {
            CSR_mat_vec_prod_parallel(q, p_buf, *row_ptr_u32_buf_ptr, *col_ind_u32_buf_ptr, val_buf, Ap_buf, n);
        } else {
            CSR_mat_vec_prod_parallel(q, p_buf, *row_ptr_buf_ptr, *col_ind_buf_ptr, val_buf, Ap_buf, n);
        }
    };
    auto cheb_apply = [&](bool init_from_r) {
        if (on_gpu) {
            apply_chebyshev_preconditioner_device(q,
                *row_ptr_u32_buf_ptr, *col_ind_u32_buf_ptr, val_buf, M_inv_buf,
                r_buf, t_buf, Ap_buf, z_buf, n, chebyshev_steps, init_from_r);
        } else {
            apply_chebyshev_preconditioner_device(q,
                *row_ptr_buf_ptr, *col_ind_buf_ptr, val_buf, M_inv_buf,
                r_buf, t_buf, Ap_buf, z_buf, n, chebyshev_steps, init_from_r);
        }
    };

    // Initial preconditioning step: Jacobi only (no steps yet).
    apply_diagonal_preconditioner(q, M_inv_buf, r_buf, z_buf, n);

    q.submit([&](sycl::handler& h) {
        auto z_acc = z_buf.get_access<sycl::access::mode::read>(h);
        auto p_acc = p_buf.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { p_acc[i] = z_acc[i]; });
    });

    double old_rr = 0.0, new_rr = 0.0, pAp = 0.0, alpha = 0.0, beta = 0.0;
    sycl::buffer<double> old_rr_buf{&old_rr, sycl::range<1>(1)};
    sycl::buffer<double> new_rr_buf{&new_rr, sycl::range<1>(1)};
    sycl::buffer<double> pAp_buf{&pAp, sycl::range<1>(1)};
    sycl::buffer<double> alpha_buf{&alpha, sycl::range<1>(1)};
    sycl::buffer<double> beta_buf{&beta, sycl::range<1>(1)};

    old_rr_buf.set_final_data(nullptr);
    new_rr_buf.set_final_data(nullptr);
    pAp_buf.set_final_data(nullptr);
    alpha_buf.set_final_data(nullptr);
    beta_buf.set_final_data(nullptr);

    scalar_product_parallel(q, r_buf, z_buf, old_rr_buf, n);

    unsigned int iteration = 0;
    unsigned int max_iter = static_cast<unsigned int>(n);
    constexpr unsigned int CHECK_STRIDE = 8;
    // Run pure Jacobi-PCG for this many iters to learn spectrum of D^{-1}A.
    // Must be >= a few times degree to capture both extremes.
    const unsigned int WARMUP_ITERS = std::min<unsigned int>(
        static_cast<unsigned int>(std::max<size_t>(24, 6 * active_degree)),
        std::max<unsigned int>(8, max_iter / 4));

    bool using_chebyshev = false;
    bool switched = false;

    // Lanczos history. alpha_hist[k-1] = alpha_k, beta_hist[k-1] = beta_k (1-indexed in math).
    std::vector<double> alpha_hist(max_iter, 0.0);
    std::vector<double> beta_hist(max_iter, 0.0);

    do {
        iteration++;
        spmv_p_to_Ap();
        scalar_product_parallel(q, p_buf, Ap_buf, pAp_buf, n);

        q.submit([&](sycl::handler& h) {
            auto old_rr_acc = old_rr_buf.get_access<sycl::access::mode::read>(h);
            auto pAp_acc = pAp_buf.get_access<sycl::access::mode::read_write>(h);
            auto alpha_acc = alpha_buf.get_access<sycl::access::mode::write>(h);
            auto new_rr_acc = new_rr_buf.get_access<sycl::access::mode::write>(h);
            h.single_task([=]() {
                alpha_acc[0] = (std::abs(pAp_acc[0]) > 1e-30) ? old_rr_acc[0] / pAp_acc[0] : 0.0;
                new_rr_acc[0] = 0.0;
                pAp_acc[0] = 0.0;
            });
        });

        // Track alpha during warmup (Lanczos of D^{-1}A). After switch we don't need it,
        // since the preconditioned operator changes and PCG-Lanczos no longer tracks D^{-1}A.
        if (!using_chebyshev) {
            q.submit([&](sycl::handler& h) {
                auto alpha_acc = alpha_buf.get_access<sycl::access::mode::read>(h);
                h.copy(alpha_acc, &alpha_hist[iteration - 1]);
            });
        }

        // Residual update kernel — variant depends on phase.
        if (using_chebyshev) {
            q.submit([&](sycl::handler& h) {
                auto x_acc = x_buf.get_access<sycl::access::mode::read_write>(h);
                auto r_acc = r_buf.get_access<sycl::access::mode::read_write>(h);
                auto p_acc = p_buf.get_access<sycl::access::mode::read>(h);
                auto Ap_acc = Ap_buf.get_access<sycl::access::mode::read>(h);
                auto alpha_acc = alpha_buf.get_access<sycl::access::mode::read>(h);
                auto M_inv_acc = M_inv_buf.get_access<sycl::access::mode::read>(h);
                auto t_acc = t_buf.get_access<sycl::access::mode::write>(h);
                auto z_acc = z_buf.get_access<sycl::access::mode::write>(h);
                auto first_tau_acc = first_tau_buf.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    x_acc[i] += alpha_acc[0] * p_acc[i];
                    const double residual = r_acc[i] - alpha_acc[0] * Ap_acc[i];
                    r_acc[i] = residual;
                    t_acc[i] = residual;
                    z_acc[i] = first_tau_acc[0] * M_inv_acc[i] * residual;
                });
            });
            cheb_apply(false);
        } else {
            q.submit([&](sycl::handler& h) {
                auto x_acc = x_buf.get_access<sycl::access::mode::read_write>(h);
                auto r_acc = r_buf.get_access<sycl::access::mode::read_write>(h);
                auto p_acc = p_buf.get_access<sycl::access::mode::read>(h);
                auto Ap_acc = Ap_buf.get_access<sycl::access::mode::read>(h);
                auto alpha_acc = alpha_buf.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    x_acc[i] += alpha_acc[0] * p_acc[i];
                    r_acc[i] = r_acc[i] - alpha_acc[0] * Ap_acc[i];
                });
            });
            apply_diagonal_preconditioner(q, M_inv_buf, r_buf, z_buf, n);
        }

        scalar_product_parallel(q, r_buf, z_buf, new_rr_buf, n);

        const bool should_check = (iteration % CHECK_STRIDE == 0) || (iteration == max_iter);
        if (should_check) {
            q.submit([&](sycl::handler& h) {
                auto acc = new_rr_buf.get_access<sycl::access::mode::read>(h);
                h.copy(acc, &new_rr);
            }).wait_and_throw();

            // Guard: r^T M^{-1} r should be > 0; negative values indicate the polynomial is
            // not SPD over the actual spectrum, so do not trust as a convergence signal.
            if ((new_rr > 0.0 && new_rr / b_norm < eps * eps) || iteration == max_iter) {
                break;
            }
        }

        q.submit([&](sycl::handler& h) {
            auto old_rr_acc = old_rr_buf.get_access<sycl::access::mode::read_write>(h);
            auto new_rr_acc = new_rr_buf.get_access<sycl::access::mode::read>(h);
            auto beta_acc = beta_buf.get_access<sycl::access::mode::write>(h);
            h.single_task([=]() {
                beta_acc[0] = (std::abs(old_rr_acc[0]) > 1e-30) ? new_rr_acc[0] / old_rr_acc[0] : 0.0;
                old_rr_acc[0] = new_rr_acc[0];
            });
        });

        if (!using_chebyshev) {
            q.submit([&](sycl::handler& h) {
                auto beta_acc = beta_buf.get_access<sycl::access::mode::read>(h);
                h.copy(beta_acc, &beta_hist[iteration - 1]);
            });
        }

        q.submit([&](sycl::handler& h) {
            auto p_acc = p_buf.get_access<sycl::access::mode::read_write>(h);
            auto z_acc = z_buf.get_access<sycl::access::mode::read>(h);
            auto beta_acc = beta_buf.get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                p_acc[i] = z_acc[i] + beta_acc[0] * p_acc[i];
            });
        });

        // End-of-warmup: build T_k from collected alpha/beta and switch to Chebyshev.
        if (!switched && iteration >= WARMUP_ITERS) {
            q.wait_and_throw(); // ensure async alpha/beta copies have landed

            const size_t k = iteration;
            std::vector<double> d(k);
            std::vector<double> e((k > 0) ? (k - 1) : 0);
            bool history_valid = true;
            for (size_t i = 0; i < k; ++i) {
                const double a = alpha_hist[i];
                if (std::abs(a) < 1e-30) { history_valid = false; break; }
                d[i] = 1.0 / a;
                if (i > 0) {
                    const double bm = beta_hist[i - 1];
                    const double am = alpha_hist[i - 1];
                    if (bm < 0.0 || std::abs(am) < 1e-30) { history_valid = false; break; }
                    d[i] += bm / am;
                    e[i - 1] = std::sqrt(std::max(bm, 0.0)) / std::abs(am);
                }
            }

            if (history_valid) {
                double new_lmin = 0.0;
                double new_lmax = 0.0;
                compute_tridiag_extreme_eigenvalues(d, e, &new_lmin, &new_lmax);

                // Lanczos extreme Ritz values are *inside* the true spectrum interval —
                // small eigenvalues are typically not yet captured after only WARMUP_ITERS
                // iterations. Pad aggressively outward so the Chebyshev polynomial stays SPD
                // over the actual spectrum (otherwise r^T M^{-1} r can go negative).
                new_lmax *= 1.20;
                new_lmin *= 0.40;

                if (new_lmin > 0.0 && new_lmax > new_lmin) {
                    lambda_min_bound = new_lmin;
                    lambda_max_bound = new_lmax;
                    build_chebyshev_steps(lambda_min_bound, lambda_max_bound, active_degree, chebyshev_steps);
                    if (!chebyshev_steps.empty()) {
                        first_tau_host = chebyshev_steps.front();
                        q.submit([&](sycl::handler& h) {
                            auto acc = first_tau_buf.get_access<sycl::access::mode::write>(h);
                            h.copy(&first_tau_host, acc);
                        });
                        using_chebyshev = true;
                        std::cout << "  [iter " << iteration << "] Lanczos bounds:"
                                  << " lambda_min=" << lambda_min_bound
                                  << " lambda_max=" << lambda_max_bound
                                  << " ratio=" << (lambda_max_bound / std::max(lambda_min_bound, 1e-30))
                                  << " -> switch to Chebyshev" << std::endl;

                        // ----- Clean CG restart with Chebyshev -----
                        // Mid-run preconditioner change breaks A-orthogonality of p and changes
                        // the scale of r^T z (the convergence test). Recompute z fresh, reset
                        // p = z, recompute old_rr, so the next iteration is "iteration 1" of
                        // a fresh Chebyshev-PCG starting from the current x.
                        cheb_apply(true);
                        q.submit([&](sycl::handler& h) {
                            auto z_acc = z_buf.get_access<sycl::access::mode::read>(h);
                            auto p_acc = p_buf.get_access<sycl::access::mode::write>(h);
                            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { p_acc[i] = z_acc[i]; });
                        });
                        // old_rr = r . z (overwrite)
                        q.submit([&](sycl::handler& h) {
                            auto acc = old_rr_buf.get_access<sycl::access::mode::write>(h);
                            h.single_task([=]() { acc[0] = 0.0; });
                        });
                        scalar_product_parallel(q, r_buf, z_buf, old_rr_buf, n);
                    } else {
                        std::cout << "  [iter " << iteration << "] Chebyshev steps empty, staying on Jacobi" << std::endl;
                    }
                } else {
                    std::cout << "  [iter " << iteration << "] Lanczos bounds invalid ("
                              << new_lmin << ", " << new_lmax << "), staying on Jacobi" << std::endl;
                }
            }
            switched = true; // do not retry
        }
    } while (iteration < max_iter);

    q.submit([&](sycl::handler& h) {
        auto acc = x_buf.get_access<sycl::access::mode::read>(h);
        h.copy(acc, x.data());
    }).wait_and_throw();
    std::cout << "CG-Chebyshev-adaptive iterations: " << iteration
              << " (warmup=" << WARMUP_ITERS
              << ", switched=" << (using_chebyshev ? "yes" : "no") << ")" << std::endl;
    return iteration;
}
