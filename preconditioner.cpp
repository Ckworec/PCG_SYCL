#include "func.h"
#include <array>

using namespace sycl;

// Forward declarations (needed for SYCL adaptation path).
static double zhukov_refine_lambda_min_from_delta(double delta,
                                                  double lambda_min_star,
                                                  double lambda_max_star,
                                                  size_t degree);
static size_t chebyshev_degree_for_accuracy(double eps1, double eta);

void compute_jacobi_preconditioner_buf(buffer<size_t>& row_ptr_buf,
                                       buffer<size_t>& diag_pos_buf,
                                       buffer<double>& val_buf,
                                       buffer<double>& M_inv_buf,
                                       size_t n,
                                       sycl::queue& q)
{
    q.submit([&](handler& h) {
        accessor val_access(val_buf, h, read_only);
        accessor diag_pos_access(diag_pos_buf, h, read_only);
        accessor M_access(M_inv_buf, h, write_only, no_init);

        h.parallel_for(range<1>(n), [=](id<1> i) {
            const size_t diag_pos = diag_pos_access[i];
            const double diag = val_access[diag_pos];
            M_access[i] = (std::abs(diag) > 1e-14) ? (1.0 / diag) : 0.0;
        });
    });
}

void build_diag_positions(const std::vector<size_t>& row_ptr,
                          const std::vector<size_t>& col_ind,
                          std::vector<size_t>& diag_pos)
{
    const size_t n = row_ptr.empty() ? 0 : row_ptr.size() - 1;
    diag_pos.assign(n, size_t(-1));

    for (size_t i = 0; i < n; ++i) {
        const size_t begin = row_ptr[i];
        const size_t end = row_ptr[i + 1];
        const auto diag_it = std::lower_bound(col_ind.begin() + begin, col_ind.begin() + end, i);
        if (diag_it == col_ind.begin() + end || *diag_it != i) {
            throw std::runtime_error("No diagonal in row " + std::to_string(i));
        }
        diag_pos[i] = static_cast<size_t>(diag_it - col_ind.begin());
    }
}

void apply_diagonal_preconditioner(sycl::queue& q,
                                   buffer<double>& M_inv_buf,
                                   buffer<double>& r_buf,
                                   buffer<double>& z_buf,
                                   size_t n)
{
    q.submit([&](handler& h) {
        accessor m_inv_acc(M_inv_buf, h, read_only);
        accessor r_acc(r_buf, h, read_only);
        accessor z_acc(z_buf, h, write_only, no_init);

        h.parallel_for(range<1>(n), [=](id<1> i) {
            z_acc[i] = m_inv_acc[i] * r_acc[i];
        });
    });
}

void compute_block_jacobi_preconditioner(const std::vector<size_t>& row_ptr,
                                         const std::vector<size_t>& col_ind,
                                         const std::vector<double>& vals,
                                         size_t n,
                                         size_t block_size,
                                         std::vector<double>& block_inv)
{
    const size_t num_blocks = (n + block_size - 1) / block_size;
    block_inv.assign(num_blocks * block_size * block_size, 0.0);

    parallel_for_host(0, num_blocks, 16, [&](size_t bi) {
        const size_t block_start = bi * block_size;
        const size_t block_end = std::min(n, block_start + block_size);
        const size_t bs = block_end - block_start;

        std::array<double, 16 * 16> B_stack{};
        std::array<double, 16 * 16> Inv_stack{};
        std::vector<double> B_dynamic;
        std::vector<double> Inv_dynamic;
        double* B = nullptr;
        double* Inv = nullptr;

        if (bs <= 16) {
            std::fill(B_stack.begin(), B_stack.end(), 0.0);
            std::fill(Inv_stack.begin(), Inv_stack.end(), 0.0);
            B = B_stack.data();
            Inv = Inv_stack.data();
        } else {
            B_dynamic.assign(bs * bs, 0.0);
            Inv_dynamic.assign(bs * bs, 0.0);
            B = B_dynamic.data();
            Inv = Inv_dynamic.data();
        }

        for (size_t i = 0; i < bs; ++i) {
            Inv[i * bs + i] = 1.0;
            size_t row = block_start + i;
            for (size_t ptr = row_ptr[row]; ptr < row_ptr[row + 1]; ++ptr) {
                size_t col = col_ind[ptr];
                if (col >= block_start && col < block_end) {
                    B[i * bs + (col - block_start)] = vals[ptr];
                }
            }
        }

        for (size_t i = 0; i < bs; ++i) {
            size_t pivot_row = i;
            double pivot = std::abs(B[i * bs + i]);
            for (size_t k = i + 1; k < bs; ++k) {
                double candidate = std::abs(B[k * bs + i]);
                if (candidate > pivot) {
                    pivot = candidate;
                    pivot_row = k;
                }
            }

            if (pivot_row != i) {
                for (size_t j = 0; j < bs; ++j) {
                    std::swap(B[i * bs + j], B[pivot_row * bs + j]);
                    std::swap(Inv[i * bs + j], Inv[pivot_row * bs + j]);
                }
            }

            double diag = B[i * bs + i];
            if (std::abs(diag) < 1e-14) {
                diag = (diag >= 0.0) ? 1e-14 : -1e-14;
                B[i * bs + i] = diag;
            }

            double inv_pivot = 1.0 / diag;
            for (size_t j = 0; j < bs; ++j) {
                B[i * bs + j] *= inv_pivot;
                Inv[i * bs + j] *= inv_pivot;
            }

            for (size_t k = 0; k < bs; ++k) {
                if (k == i) continue;
                double factor = B[k * bs + i];
                for (size_t j = 0; j < bs; ++j) {
                    B[k * bs + j] -= factor * B[i * bs + j];
                    Inv[k * bs + j] -= factor * Inv[i * bs + j];
                }
            }
        }

        for (size_t i = 0; i < bs; ++i) {
            for (size_t j = 0; j < bs; ++j) {
                block_inv[bi * block_size * block_size + i * block_size + j] = Inv[i * bs + j];
            }
        }
    });
}

static size_t choose_block_jacobi_local_size(const sycl::queue& q, size_t block_size)
{
    const size_t max_work_group_size =
        q.get_device().get_info<sycl::info::device::max_work_group_size>();
    return std::max<size_t>(1, std::min(block_size, max_work_group_size));
}

template <size_t BlockSize, size_t BlocksPerGroup>
static void compute_block_jacobi_preconditioner_device_fixed(sycl::queue& q,
                                                             sycl::buffer<size_t>& row_ptr_buf,
                                                             sycl::buffer<size_t>& col_ind_buf,
                                                             sycl::buffer<double>& val_buf,
                                                             sycl::buffer<double>& block_inv_buf,
                                                             size_t n)
{
    const size_t num_blocks = (n + BlockSize - 1) / BlockSize;
    const size_t num_groups = (num_blocks + BlocksPerGroup - 1) / BlocksPerGroup;
    constexpr size_t local_size = BlockSize * BlocksPerGroup;
    constexpr size_t block_stride = BlockSize * BlockSize;

    q.submit([&](sycl::handler& h) {
        sycl::accessor row_ptr(row_ptr_buf, h, sycl::read_only);
        sycl::accessor col_ind(col_ind_buf, h, sycl::read_only);
        sycl::accessor vals(val_buf, h, sycl::read_only);
        sycl::accessor block_inv(block_inv_buf, h, sycl::write_only, sycl::no_init);
        sycl::local_accessor<double, 1> block_mat_local(sycl::range<1>(BlocksPerGroup * block_stride), h);
        sycl::local_accessor<double, 1> block_inv_local(sycl::range<1>(BlocksPerGroup * block_stride), h);
        sycl::local_accessor<size_t, 1> pivot_row_local(sycl::range<1>(BlocksPerGroup), h);
        sycl::local_accessor<double, 1> pivot_diag_local(sycl::range<1>(BlocksPerGroup), h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_groups * local_size),
                                         sycl::range<1>(local_size)),
            [=](sycl::nd_item<1> item) {
                const size_t lid = item.get_local_id(0);
                const size_t segment = lid / BlockSize;
                const size_t lane = lid % BlockSize;
                const size_t block_idx = item.get_group(0) * BlocksPerGroup + segment;
                const size_t local_block_offset = segment * block_stride;

                for (size_t idx = lane; idx < block_stride; idx += BlockSize) {
                    block_mat_local[local_block_offset + idx] = 0.0;
                    block_inv_local[local_block_offset + idx] = 0.0;
                }
                item.barrier(sycl::access::fence_space::local_space);

                size_t bs = 0;
                size_t global_offset = 0;
                size_t block_start = 0;
                if (block_idx < num_blocks) {
                    block_start = block_idx * BlockSize;
                    const size_t block_end = sycl::min(block_start + BlockSize, n);
                    bs = block_end - block_start;
                    global_offset = block_idx * block_stride;
                }

                if (block_idx < num_blocks && lane < bs) {
                    block_inv_local[local_block_offset + lane * BlockSize + lane] = 1.0;
                    const size_t row = block_start + lane;
                    for (size_t ptr = row_ptr[row]; ptr < row_ptr[row + 1]; ++ptr) {
                        const size_t col = col_ind[ptr];
                        if (col >= block_start && col < block_start + bs) {
                            block_mat_local[local_block_offset + lane * BlockSize + (col - block_start)] = vals[ptr];
                        }
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (size_t i = 0; i < bs; ++i) {
                    if (lane == 0 && block_idx < num_blocks) {
                        size_t pivot_row = i;
                        double pivot = sycl::fabs(block_mat_local[local_block_offset + i * BlockSize + i]);
                        for (size_t k = i + 1; k < bs; ++k) {
                            const double candidate =
                                sycl::fabs(block_mat_local[local_block_offset + k * BlockSize + i]);
                            if (candidate > pivot) {
                                pivot = candidate;
                                pivot_row = k;
                            }
                        }
                        pivot_row_local[segment] = pivot_row;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (block_idx < num_blocks) {
                        const size_t pivot_row = pivot_row_local[segment];
                        if (pivot_row != i) {
                            for (size_t j = lane; j < bs; j += BlockSize) {
                                const size_t row_i = local_block_offset + i * BlockSize + j;
                                const size_t row_p = local_block_offset + pivot_row * BlockSize + j;

                                const double tmp_mat = block_mat_local[row_i];
                                block_mat_local[row_i] = block_mat_local[row_p];
                                block_mat_local[row_p] = tmp_mat;

                                const double tmp_inv = block_inv_local[row_i];
                                block_inv_local[row_i] = block_inv_local[row_p];
                                block_inv_local[row_p] = tmp_inv;
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (lane == 0 && block_idx < num_blocks) {
                        double diag = block_mat_local[local_block_offset + i * BlockSize + i];
                        if (sycl::fabs(diag) < 1e-14) {
                            diag = (diag >= 0.0) ? 1e-14 : -1e-14;
                            block_mat_local[local_block_offset + i * BlockSize + i] = diag;
                        }
                        pivot_diag_local[segment] = diag;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (block_idx < num_blocks) {
                        const double inv_pivot = 1.0 / pivot_diag_local[segment];
                        for (size_t j = lane; j < bs; j += BlockSize) {
                            block_mat_local[local_block_offset + i * BlockSize + j] *= inv_pivot;
                            block_inv_local[local_block_offset + i * BlockSize + j] *= inv_pivot;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (block_idx < num_blocks && lane < bs && lane != i) {
                        const size_t row_k = lane;
                        const double factor = block_mat_local[local_block_offset + row_k * BlockSize + i];
                        if (sycl::fabs(factor) > 1e-30) {
                            for (size_t j = 0; j < bs; ++j) {
                                block_mat_local[local_block_offset + row_k * BlockSize + j] -=
                                    factor * block_mat_local[local_block_offset + i * BlockSize + j];
                                block_inv_local[local_block_offset + row_k * BlockSize + j] -=
                                    factor * block_inv_local[local_block_offset + i * BlockSize + j];
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (block_idx < num_blocks && lane < BlockSize) {
                    for (size_t j = 0; j < BlockSize; ++j) {
                        const double value = (lane < bs && j < bs)
                            ? block_inv_local[local_block_offset + lane * BlockSize + j]
                            : 0.0;
                        block_inv[global_offset + lane * BlockSize + j] = value;
                    }
                }
            });
    });
}

template <size_t BlockSize, size_t BlocksPerGroup>
static void apply_block_jacobi_preconditioner_device_fixed(sycl::queue& q,
                                                           buffer<double>& block_inv_buf,
                                                           buffer<double>& r_buf,
                                                           buffer<double>& z_buf,
                                                           size_t n)
{
    const size_t num_blocks = (n + BlockSize - 1) / BlockSize;
    const size_t num_groups = (num_blocks + BlocksPerGroup - 1) / BlocksPerGroup;
    constexpr size_t local_size = BlockSize * BlocksPerGroup;
    constexpr size_t block_stride = BlockSize * BlockSize;

    q.submit([&](handler& h) {
        accessor m_inv_acc(block_inv_buf, h, read_only);
        accessor r_acc(r_buf, h, read_only);
        accessor z_acc(z_buf, h, write_only, no_init);
        sycl::local_accessor<double, 1> r_local(sycl::range<1>(BlocksPerGroup * BlockSize), h);
        // Cache the per-block inverse in local memory: stride-1 coalesced loads
        // from global, then the matvec reads from local with no bank conflicts
        // for typical BlockSize (4, 8, 16).
        sycl::local_accessor<double, 1> m_inv_local(
            sycl::range<1>(BlocksPerGroup * block_stride), h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_groups * local_size),
                                         sycl::range<1>(local_size)),
            [=](sycl::nd_item<1> item) {
                const size_t lid = item.get_local_id(0);
                const size_t segment = lid / BlockSize;
                const size_t lane = lid % BlockSize;
                const size_t block_idx = item.get_group(0) * BlocksPerGroup + segment;
                const size_t local_rhs_offset = segment * BlockSize;
                const size_t local_block_offset = segment * block_stride;

                size_t bs = 0;
                size_t block_start = 0;
                if (block_idx < num_blocks) {
                    block_start = block_idx * BlockSize;
                    bs = sycl::min(BlockSize, n - block_start);
                }

                // Coalesced load of the entire block_inv tile into local memory.
                // Each lane streams stride-1 through its slice of the tile.
                if (block_idx < num_blocks) {
                    const size_t global_block_offset = block_idx * block_stride;
                    for (size_t idx = lane; idx < block_stride; idx += BlockSize) {
                        m_inv_local[local_block_offset + idx] = m_inv_acc[global_block_offset + idx];
                    }
                }

                if (block_idx < num_blocks && lane < bs) {
                    r_local[local_rhs_offset + lane] = r_acc[block_start + lane];
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (block_idx < num_blocks && lane < bs) {
                    const size_t local_off = local_block_offset + lane * BlockSize;
                    double sum = 0.0;
                    for (size_t j = 0; j < bs; ++j) {
                        sum += m_inv_local[local_off + j] * r_local[local_rhs_offset + j];
                    }
                    z_acc[block_start + lane] = sum;
                }
            });
    });
}

void compute_block_jacobi_preconditioner_device(sycl::queue& q,
                                                sycl::buffer<size_t>& row_ptr_buf,
                                                sycl::buffer<size_t>& col_ind_buf,
                                                sycl::buffer<double>& val_buf,
                                                sycl::buffer<double>& block_inv_buf,
                                                size_t n,
                                                size_t block_size)
{
    if (q.get_device().is_gpu()) {
        switch (block_size) {
            case 4:
                compute_block_jacobi_preconditioner_device_fixed<4, 8>(
                    q, row_ptr_buf, col_ind_buf, val_buf, block_inv_buf, n);
                return;
            case 7:
                compute_block_jacobi_preconditioner_device_fixed<7, 4>(
                    q, row_ptr_buf, col_ind_buf, val_buf, block_inv_buf, n);
                return;
            case 8:
                compute_block_jacobi_preconditioner_device_fixed<8, 4>(
                    q, row_ptr_buf, col_ind_buf, val_buf, block_inv_buf, n);
                return;
            case 16:
                compute_block_jacobi_preconditioner_device_fixed<16, 2>(
                    q, row_ptr_buf, col_ind_buf, val_buf, block_inv_buf, n);
                return;
            default:
                break;
        }
    }

    const size_t num_blocks = (n + block_size - 1) / block_size;
    const size_t local_size = choose_block_jacobi_local_size(q, block_size);
    const size_t block_stride = block_size * block_size;

    q.submit([&](sycl::handler& h) {
        sycl::accessor row_ptr(row_ptr_buf, h, sycl::read_only);
        sycl::accessor col_ind(col_ind_buf, h, sycl::read_only);
        sycl::accessor vals(val_buf, h, sycl::read_only);
        sycl::accessor block_inv(block_inv_buf, h, sycl::write_only, sycl::no_init);
        sycl::local_accessor<double, 1> block_mat_local(sycl::range<1>(block_stride), h);
        sycl::local_accessor<double, 1> block_inv_local(sycl::range<1>(block_stride), h);
        sycl::local_accessor<size_t, 1> pivot_row_local(sycl::range<1>(1), h);
        sycl::local_accessor<double, 1> pivot_diag_local(sycl::range<1>(1), h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_blocks * local_size),
                                         sycl::range<1>(local_size)),
            [=](sycl::nd_item<1> item) {
                const size_t bi = item.get_group(0);
                const size_t lid = item.get_local_id(0);
                const size_t block_start = bi * block_size;
                const size_t block_end = sycl::min(block_start + block_size, n);
                const size_t bs = block_end - block_start;
                const size_t global_offset = bi * block_stride;

                for (size_t idx = lid; idx < block_stride; idx += local_size) {
                    block_mat_local[idx] = 0.0;
                    block_inv_local[idx] = 0.0;
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (lid < bs) {
                    block_inv_local[lid * block_size + lid] = 1.0;
                    const size_t row = block_start + lid;
                    for (size_t ptr = row_ptr[row]; ptr < row_ptr[row + 1]; ++ptr) {
                        const size_t col = col_ind[ptr];
                        if (col >= block_start && col < block_end) {
                            block_mat_local[lid * block_size + (col - block_start)] = vals[ptr];
                        }
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (size_t i = 0; i < bs; ++i) {
                    if (lid == 0) {
                        size_t pivot_row = i;
                        double pivot = sycl::fabs(block_mat_local[i * block_size + i]);
                        for (size_t k = i + 1; k < bs; ++k) {
                            const double candidate = sycl::fabs(block_mat_local[k * block_size + i]);
                            if (candidate > pivot) {
                                pivot = candidate;
                                pivot_row = k;
                            }
                        }
                        pivot_row_local[0] = pivot_row;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    const size_t pivot_row = pivot_row_local[0];
                    if (pivot_row != i) {
                        for (size_t j = lid; j < bs; j += local_size) {
                            const size_t row_i = i * block_size + j;
                            const size_t row_p = pivot_row * block_size + j;

                            const double tmp_mat = block_mat_local[row_i];
                            block_mat_local[row_i] = block_mat_local[row_p];
                            block_mat_local[row_p] = tmp_mat;

                            const double tmp_inv = block_inv_local[row_i];
                            block_inv_local[row_i] = block_inv_local[row_p];
                            block_inv_local[row_p] = tmp_inv;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (lid == 0) {
                        double diag = block_mat_local[i * block_size + i];
                        if (sycl::fabs(diag) < 1e-14) {
                            diag = (diag >= 0.0) ? 1e-14 : -1e-14;
                            block_mat_local[i * block_size + i] = diag;
                        }
                        pivot_diag_local[0] = diag;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    const double inv_pivot = 1.0 / pivot_diag_local[0];
                    for (size_t j = lid; j < bs; j += local_size) {
                        block_mat_local[i * block_size + j] *= inv_pivot;
                        block_inv_local[i * block_size + j] *= inv_pivot;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (lid < bs && lid != i) {
                        const size_t row_k = lid;
                        const double factor = block_mat_local[row_k * block_size + i];
                        if (sycl::fabs(factor) > 1e-30) {
                            for (size_t j = 0; j < bs; ++j) {
                                block_mat_local[row_k * block_size + j] -=
                                    factor * block_mat_local[i * block_size + j];
                                block_inv_local[row_k * block_size + j] -=
                                    factor * block_inv_local[i * block_size + j];
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (lid < bs) {
                    for (size_t j = 0; j < bs; ++j) {
                        block_inv[global_offset + lid * block_size + j] =
                            block_inv_local[lid * block_size + j];
                    }
                    for (size_t j = bs; j < block_size; ++j) {
                        block_inv[global_offset + lid * block_size + j] = 0.0;
                    }
                }

                if (lid >= bs) {
                    for (size_t j = 0; j < block_size; ++j) {
                        block_inv[global_offset + lid * block_size + j] = 0.0;
                    }
                }
            });
    });
}

void apply_block_jacobi_preconditioner(sycl::queue& q,
                                       buffer<double>& block_inv_buf,
                                       buffer<double>& r_buf,
                                       buffer<double>& z_buf,
                                       size_t n,
                                       size_t block_size)
{
    if (q.get_device().is_gpu()) {
        switch (block_size) {
            case 4:
                apply_block_jacobi_preconditioner_device_fixed<4, 8>(q, block_inv_buf, r_buf, z_buf, n);
                return;
            case 7:
                apply_block_jacobi_preconditioner_device_fixed<7, 4>(q, block_inv_buf, r_buf, z_buf, n);
                return;
            case 8:
                apply_block_jacobi_preconditioner_device_fixed<8, 4>(q, block_inv_buf, r_buf, z_buf, n);
                return;
            case 16:
                apply_block_jacobi_preconditioner_device_fixed<16, 2>(q, block_inv_buf, r_buf, z_buf, n);
                return;
            default:
                break;
        }
    }

    const size_t num_blocks = (n + block_size - 1) / block_size;
    const size_t local_size = choose_block_jacobi_local_size(q, block_size);

    q.submit([&](handler& h) {
        accessor m_inv_acc(block_inv_buf, h, read_only);
        accessor r_acc(r_buf, h, read_only);
        accessor z_acc(z_buf, h, write_only, no_init);
        sycl::local_accessor<double, 1> r_local(sycl::range<1>(block_size), h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_blocks * local_size),
                                         sycl::range<1>(local_size)),
            [=](sycl::nd_item<1> item) {
                const size_t block_idx = item.get_group(0);
                const size_t lid = item.get_local_id(0);
                const size_t block_start = block_idx * block_size;
                const size_t bs = sycl::min(block_size, n - block_start);

                if (lid < bs) {
                    r_local[lid] = r_acc[block_start + lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (lid < bs) {
                    const size_t offset = block_idx * block_size * block_size + lid * block_size;
                    double sum = 0.0;
                    for (size_t j = 0; j < bs; ++j) {
                        sum += m_inv_acc[offset + j] * r_local[j];
                    }
                    z_acc[block_start + lid] = sum;
                }
            });
    });
}

void compute_spai_preconditioner_buf(buffer<size_t>& row_ptr_buf,
                                     buffer<size_t>& diag_pos_buf,
                                     buffer<double>& val_buf,
                                     buffer<double>& M_inv_buf,
                                     size_t n,
                                     sycl::queue& q)
{
    q.submit([&](handler& h) {
        accessor row_ptr_acc(row_ptr_buf, h, read_only);
        accessor diag_pos_acc(diag_pos_buf, h, read_only);
        accessor val_acc(val_buf, h, read_only);
        accessor M_acc(M_inv_buf, h, write_only, no_init);

        h.parallel_for(range<1>(n), [=](id<1> i) {
            const double diag = val_acc[diag_pos_acc[i]];
            double row_sq_sum = 0.0;
            for (size_t k = row_ptr_acc[i]; k < row_ptr_acc[i + 1]; ++k) {
                const double a_ik = val_acc[k];
                row_sq_sum += a_ik * a_ik;
            }

            if (std::abs(diag) > 1e-14 && row_sq_sum > 1e-28) {
                M_acc[i] = diag / row_sq_sum;
            } else {
                M_acc[i] = 0.0;
            }
        });
    });
}

// Fused variant: finds the diagonal entry inline in one pass, removing the
// host-side build_diag_positions() step plus one device buffer.
void compute_spai_preconditioner_inline_buf(buffer<size_t>& row_ptr_buf,
                                            buffer<size_t>& col_ind_buf,
                                            buffer<double>& val_buf,
                                            buffer<double>& M_inv_buf,
                                            size_t n,
                                            sycl::queue& q)
{
    q.submit([&](handler& h) {
        accessor row_ptr_acc(row_ptr_buf, h, read_only);
        accessor col_ind_acc(col_ind_buf, h, read_only);
        accessor val_acc(val_buf, h, read_only);
        accessor M_acc(M_inv_buf, h, write_only, no_init);

        h.parallel_for(range<1>(n), [=](id<1> i) {
            const size_t row = i[0];
            const size_t begin = row_ptr_acc[row];
            const size_t end = row_ptr_acc[row + 1];
            double row_sq_sum = 0.0;
            double diag = 0.0;
            for (size_t k = begin; k < end; ++k) {
                const double a_ik = val_acc[k];
                row_sq_sum += a_ik * a_ik;
                if (col_ind_acc[k] == row) {
                    diag = a_ik;
                }
            }

            if (sycl::fabs(diag) > 1e-14 && row_sq_sum > 1e-28) {
                M_acc[row] = diag / row_sq_sum;
            } else {
                M_acc[row] = 0.0;
            }
        });
    });
}

static double vector_norm_sq_host(const std::vector<double>& x)
{
    return parallel_sum_host<double>(0, x.size(), 16384, 0.0, [&](size_t i) {
        return x[i] * x[i];
    });
}

static void csr_mat_vec_prod_host_local(const std::vector<size_t>& row_ptr,
                                        const std::vector<size_t>& col_ind,
                                        const std::vector<double>& vals,
                                        const std::vector<double>& x,
                                        std::vector<double>& y)
{
    const size_t n = x.size();
    y.resize(n);
    parallel_for_host(0, n, 16384, [&](size_t i) {
        double sum = 0.0;
        for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            sum += vals[k] * x[col_ind[k]];
        }
        y[i] = sum;
    });
}

void estimate_gershgorin_upper_bound_device(sycl::queue& q,
                                            sycl::buffer<size_t>& row_ptr_buf,
                                            sycl::buffer<double>& val_buf,
                                            sycl::buffer<double>& row_sums_buf,
                                            sycl::buffer<double>& lambda_max_buf,
                                            size_t n)
{
    q.submit([&](sycl::handler& h) {
        sycl::accessor row_ptr(row_ptr_buf, h, sycl::read_only);
        sycl::accessor val(val_buf, h, sycl::read_only);
        sycl::accessor out(row_sums_buf, h, sycl::write_only, sycl::no_init);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            double s = 0.0;
            for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                s += sycl::fabs(val[k]);
            }
            out[i] = s;
        });
    });

    q.submit([&](sycl::handler& h) {
        sycl::accessor lambda_max(lambda_max_buf, h, sycl::write_only, sycl::no_init);
        h.single_task([=]() {
            lambda_max[0] = 0.0;
        });
    });

    q.submit([&](sycl::handler& h) {
        sycl::accessor in(row_sums_buf, h, sycl::read_only);
        auto red = sycl::reduction(lambda_max_buf, h, sycl::maximum<double>());
        h.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> i, auto& m) {
            m.combine(in[i]);
        });
    });
}

void estimate_chebyshev_lambda_min_device(sycl::queue& q,
                                          sycl::buffer<size_t>& row_ptr_buf,
                                          sycl::buffer<size_t>& col_ind_buf,
                                          sycl::buffer<double>& val_buf,
                                          sycl::buffer<double>& rhs_buf,
                                          sycl::buffer<double>& probe_buf,
                                          sycl::buffer<double>& Ap_buf,
                                          sycl::buffer<double>& norm_buf,
                                          sycl::buffer<double>& rayleigh_buf,
                                          sycl::buffer<double>& lambda_max_buf,
                                          sycl::buffer<double>& lambda_min_buf,
                                          size_t n)
{
    q.submit([&](sycl::handler& h) {
        sycl::accessor rhs(rhs_buf, h, sycl::read_only);
        sycl::accessor probe(probe_buf, h, sycl::write_only, sycl::no_init);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            probe[i] = rhs[i];
        });
    });

    q.submit([&](sycl::handler& h) {
        sycl::accessor norm(norm_buf, h, sycl::write_only, sycl::no_init);
        h.single_task([=]() {
            norm[0] = 0.0;
        });
    });
    
    scalar_product_parallel(q, probe_buf, probe_buf, norm_buf, n);

    q.submit([&](sycl::handler& h) {
        sycl::accessor norm(norm_buf, h, sycl::read_only);
        sycl::accessor probe(probe_buf, h, sycl::read_write);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            if (norm[0] < 1e-20) {
                probe[i] = 1.0;
            }
        });
    });

    q.submit([&](sycl::handler& h) {
        sycl::accessor norm(norm_buf, h, sycl::write_only, sycl::no_init);
        sycl::accessor rayleigh(rayleigh_buf, h, sycl::write_only, sycl::no_init);
        h.single_task([=]() {
            norm[0] = 0.0;
            rayleigh[0] = 0.0;
        });
    });

    scalar_product_parallel(q, probe_buf, probe_buf, norm_buf, n);
    CSR_mat_vec_prod_parallel(q, probe_buf, row_ptr_buf, col_ind_buf, val_buf, Ap_buf, n);
    scalar_product_parallel(q, Ap_buf, probe_buf, rayleigh_buf, n);

    q.submit([&](sycl::handler& h) {
        sycl::accessor norm(norm_buf, h, sycl::read_only);
        sycl::accessor rayleigh(rayleigh_buf, h, sycl::read_only);
        sycl::accessor lambda_max(lambda_max_buf, h, sycl::read_only);
        sycl::accessor lambda_min(lambda_min_buf, h, sycl::write_only, sycl::no_init);
        h.single_task([=]() {
            const double lambda_max_value = lambda_max[0];
            if (lambda_max_value <= 1e-14) {
                lambda_min[0] = std::max(1e-14, 1e-6 * lambda_max_value);
                return;
            }

            const double denom = sycl::fmax(norm[0], 1e-30);
            double lambda_min_value = rayleigh[0] / denom;
            lambda_min_value = sycl::fmax(lambda_min_value, 1e-14);
            lambda_min_value = sycl::fmin(lambda_min_value, 0.999999 * lambda_max_value);
            lambda_min[0] = lambda_min_value;
        });
    });
}

// Оператор F_p(A) = ∏_k (I - τ_k A) из препринта ИПМ 2018-172, (3); нужен для δ = ||F_p(A)v||/||v||.
static void apply_chebyshev_operator_product_host(const std::vector<size_t>& row_ptr,
                                                  const std::vector<size_t>& col_ind,
                                                  const std::vector<double>& vals,
                                                  const std::vector<double>& r,
                                                  std::vector<double>& t,
                                                  std::vector<double>& w,
                                                  const std::vector<double>& steps)
{
    const size_t n = r.size();
    t = r;

    for (double tau : steps) {
        csr_mat_vec_prod_host_local(row_ptr, col_ind, vals, t, w);

        for (size_t i = 0; i < n; ++i) {
            t[i] -= tau * w[i];
        }
    }
}

// Предобуславливатель в CG: прежняя (стабильная в расчётах) схема накопления.
static void apply_chebyshev_host(const std::vector<size_t>& row_ptr,
                                 const std::vector<size_t>& col_ind,
                                 const std::vector<double>& vals,
                                 const std::vector<double>& r,
                                 std::vector<double>& z,
                                 std::vector<double>& t,
                                 std::vector<double>& w,
                                 const std::vector<double>& steps)
{
    const size_t n = r.size();
    t = r;
    z.assign(n, 0.0);

    for (double tau : steps) {
        csr_mat_vec_prod_host_local(row_ptr, col_ind, vals, t, w);

        parallel_for_host(0, n, 512, [&](size_t i) {
            const double current_t = t[i];
            z[i] += tau * current_t;
            t[i] = current_t - tau * w[i];
        });
    }
}

// Упорядочивание параметров τ (устойчивость по Лебедеву–Финогенову): чередование «с краёв».
static void permute_chebyshev_tau_order(size_t degree, std::vector<double>& tau)
{
    if (degree <= 1) {
        return;
    }
    std::vector<double> natural = tau;
    for (size_t j = 0; j < degree; ++j) {
        const size_t idx = (j % 2 == 0) ? (j / 2) : (degree - 1 - j / 2);
        tau[j] = natural[idx];
    }
}

// Нули β_k многочлена T_p на [-1,1], формула (7); θ_k = d - c β_k; τ_k = 1/θ_k, формула (8) препринта 2018-172.
void build_chebyshev_steps(double lambda_min_estimate,
                           double lambda_max_estimate,
                           size_t degree,
                           std::vector<double>& steps)
{
    steps.clear();
    if (degree == 0 || lambda_max_estimate <= 1e-14) {
        return;
    }

    lambda_min_estimate = std::clamp(lambda_min_estimate, 1e-14, 0.999999 * lambda_max_estimate);

    const double d = 0.5 * (lambda_max_estimate + lambda_min_estimate);
    const double c = 0.5 * (lambda_max_estimate - lambda_min_estimate);
    const double pi = std::acos(-1.0);

    steps.resize(degree);
    for (size_t k = 0; k < degree; ++k) {
        const double beta = std::cos(pi * (2.0 * static_cast<double>(k) + 1.0) / (2.0 * static_cast<double>(degree)));
        const double theta = d - c * beta;
        steps[k] = (std::abs(theta) > 1e-30) ? 1.0 / theta : 0.0;
    }
    permute_chebyshev_tau_order(degree, steps);
}

// Формула (9) препринта 2018-172: число итераций Чебышёва для достижения точности eps1
// при η = λ*_min / λ*_max. Используется только для адаптации (не для предобуславливателя CG).
static size_t chebyshev_degree_for_accuracy(double eps1, double eta)
{
    eta = std::clamp(eta, 1e-12, 0.999999);
    const double sqrt_eta = std::sqrt(eta);
    // ln((1+√η)/(1-√η))
    const double log_rho = std::log((1.0 + sqrt_eta) / (1.0 - sqrt_eta));
    if (log_rho < 1e-20) return 200;
    eps1 = std::clamp(eps1, 1e-15, 0.999);
    const double inv_eps = 1.0 / eps1;
    // ln(ε⁻¹ + √(ε⁻² - 1))
    const double numer = std::log(inv_eps + std::sqrt(inv_eps * inv_eps - 1.0));
    const size_t p = static_cast<size_t>(std::ceil(numer / log_rho));
    return std::clamp(p, size_t(1), size_t(200));
}

// Препринт ИПМ 2018-172 (Жуков, Новикова, Феодоритова), п.4: уточнение λ*_min по δ = ||r_p||/||r_0|| и формулы (6).
static double zhukov_refine_lambda_min_from_delta(double delta,
                                                double lambda_min_star,
                                                double lambda_max_star,
                                                size_t degree)
{
    if (!std::isfinite(delta) || degree == 0) {
        return lambda_min_star;
    }

    double eta = std::clamp(lambda_min_star / lambda_max_star, 1e-30, 0.999999);
    const double sqrt_eta = std::sqrt(eta);
    const double rho = (1.0 - sqrt_eta) / std::max(1e-30, 1.0 + sqrt_eta);
    const double p = static_cast<double>(degree);
    const double qp = 2.0 / (std::pow(rho, p) + std::pow(rho, -p));
    const double y1 = delta / std::max(qp, 1e-300);

    if (y1 <= 1.0) {
        return lambda_min_star;
    }

    const double x_star = std::cosh(std::acosh(y1) / p);
    double lambda_new = lambda_max_star * (0.5 * (1.0 + eta) - 0.5 * (1.0 - eta) * x_star);

    if (!std::isfinite(lambda_new)) {
        return lambda_min_star;
    }

    const double upper =
        std::min(0.999999 * lambda_max_star, std::max(2e-14, 0.999999 * lambda_min_star));
    return std::clamp(lambda_new, 1e-14, upper);
}

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
                                           bool initialize_from_r)
{
    if (steps.empty()) {
        apply_diagonal_preconditioner(q, M_inv_buf, r_buf, z_buf, n);
        return;
    }

    if (initialize_from_r) {
        const double first_tau = steps.front();
        q.submit([&](sycl::handler& h) {
            sycl::accessor r(r_buf, h, sycl::read_only);
            sycl::accessor t(t_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor M_inv(M_inv_buf, h, sycl::read_only);
            sycl::accessor z(z_buf, h, sycl::write_only, sycl::no_init);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                const double residual = r[i];
                t[i] = residual;
                z[i] = first_tau * M_inv[i] * residual;
            });
        });
    }
    else if (steps.size() <= 1) {
        return;
    }

    sycl::buffer<double>* current_t_buf = &t_buf;
    sycl::buffer<double>* next_t_buf = &tmp_buf;

    for (size_t step = 1; step < steps.size(); ++step) {
        const double prev_tau = steps[step - 1];
        const double tau = steps[step];

        q.submit([&](sycl::handler& h) {
            sycl::accessor row_ptr(row_ptr_buf, h, sycl::read_only);
            sycl::accessor col_ind(col_ind_buf, h, sycl::read_only);
            sycl::accessor val(val_buf, h, sycl::read_only);
            sycl::accessor M_inv(M_inv_buf, h, sycl::read_only);
            sycl::accessor current_t(*current_t_buf, h, sycl::read_only);
            sycl::accessor next_t(*next_t_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor z(z_buf, h, sycl::read_write);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                const size_t row = i[0];
                double scaled_A_t = 0.0;
                for (size_t k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
                    const size_t col = col_ind[k];
                    scaled_A_t += val[k] * M_inv[col] * current_t[col];
                }

                const double next_t_value = current_t[row] - prev_tau * scaled_A_t;
                next_t[row] = next_t_value;
                z[row] += tau * M_inv[row] * next_t_value;
            });
        });

        std::swap(current_t_buf, next_t_buf);
    }
}

// 32-bit-index overload — saves SpMV bandwidth on GPU.
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
                                           bool initialize_from_r)
{
    if (steps.empty()) {
        apply_diagonal_preconditioner(q, M_inv_buf, r_buf, z_buf, n);
        return;
    }

    if (initialize_from_r) {
        const double first_tau = steps.front();
        q.submit([&](sycl::handler& h) {
            sycl::accessor r(r_buf, h, sycl::read_only);
            sycl::accessor t(t_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor M_inv(M_inv_buf, h, sycl::read_only);
            sycl::accessor z(z_buf, h, sycl::write_only, sycl::no_init);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                const double residual = r[i];
                t[i] = residual;
                z[i] = first_tau * M_inv[i] * residual;
            });
        });
    }
    else if (steps.size() <= 1) {
        return;
    }

    sycl::buffer<double>* current_t_buf = &t_buf;
    sycl::buffer<double>* next_t_buf = &tmp_buf;

    for (size_t step = 1; step < steps.size(); ++step) {
        const double prev_tau = steps[step - 1];
        const double tau = steps[step];

        q.submit([&](sycl::handler& h) {
            sycl::accessor row_ptr(row_ptr_buf, h, sycl::read_only);
            sycl::accessor col_ind(col_ind_buf, h, sycl::read_only);
            sycl::accessor val(val_buf, h, sycl::read_only);
            sycl::accessor M_inv(M_inv_buf, h, sycl::read_only);
            sycl::accessor current_t(*current_t_buf, h, sycl::read_only);
            sycl::accessor next_t(*next_t_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor z(z_buf, h, sycl::read_write);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                const uint32_t row = static_cast<uint32_t>(i[0]);
                double scaled_A_t = 0.0;
                const uint32_t begin = row_ptr[row];
                const uint32_t end = row_ptr[row + 1];
                for (uint32_t k = begin; k < end; ++k) {
                    const uint32_t col = col_ind[k];
                    scaled_A_t += val[k] * M_inv[col] * current_t[col];
                }

                const double next_t_value = current_t[row] - prev_tau * scaled_A_t;
                next_t[row] = next_t_value;
                z[row] += tau * M_inv[row] * next_t_value;
            });
        });

        std::swap(current_t_buf, next_t_buf);
    }
}

template <typename Func>
static void parallel_for_chunks(size_t begin, size_t end, Func&& fn)
{
    if (end <= begin) {
        return;
    }

    const size_t work_size = end - begin;
    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t thread_count = std::min<size_t>(hw_threads, work_size);

    if (thread_count <= 1 || work_size < 256) {
        for (size_t idx = begin; idx < end; ++idx) {
            fn(idx);
        }
        return;
    }

    const size_t chunk = (work_size + thread_count - 1) / thread_count;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    for (size_t thread_id = 0; thread_id < thread_count; ++thread_id) {
        const size_t chunk_begin = begin + thread_id * chunk;
        const size_t chunk_end = std::min(end, chunk_begin + chunk);
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

static void build_level_schedule(const std::vector<size_t>& levels,
                                 std::vector<size_t>& level_ptr,
                                 std::vector<size_t>& rows)
{
    const size_t n = levels.size();
    const size_t max_level = levels.empty() ? 0 : *std::max_element(levels.begin(), levels.end());

    level_ptr.assign(max_level + 2, 0);
    for (size_t level : levels) {
        level_ptr[level + 1]++;
    }
    for (size_t i = 1; i < level_ptr.size(); ++i) {
        level_ptr[i] += level_ptr[i - 1];
    }

    rows.assign(n, 0);
    std::vector<size_t> offsets = level_ptr;
    for (size_t row = 0; row < n; ++row) {
        rows[offsets[levels[row]]++] = row;
    }
}

void ic0_factor(size_t n,
                const std::vector<size_t>& row_ptr,
                const std::vector<size_t>& col_idx,
                const std::vector<double>& vals,
                IC0Preconditioner& ic0)
{
    const size_t not_found = size_t(-1);
    const double relative_shift = 1e-1;
    const double absolute_shift = 1e-14;
    ic0.row_ptr = row_ptr;
    ic0.col_idx = col_idx;
    ic0.perm.resize(n);
    ic0.inv_perm.resize(n);
    for (size_t i = 0; i < n; ++i) {
        ic0.perm[i] = i;
        ic0.inv_perm[i] = i;
    }

    ic0.L_vals.assign(vals.size(), 0.0);
    ic0.diag.assign(n, 0.0);
    ic0.diag_pos.assign(n, not_found);

    std::vector<size_t> forward_levels(n, 0);

    for (size_t i = 0; i < n; ++i) {
        size_t level = 0;
        for (size_t idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            const size_t col = col_idx[idx];
            if (col == i) {
                ic0.diag_pos[i] = idx;
                break;
            }
            if (col < i) {
                level = std::max(level, forward_levels[col] + 1);
            }
        }

        if (ic0.diag_pos[i] == not_found) {
            throw std::runtime_error("No diagonal in row " + std::to_string(i));
        }

        if (!(vals[ic0.diag_pos[i]] > 0.0) || !std::isfinite(vals[ic0.diag_pos[i]])) {
            throw std::runtime_error("IC0 requires positive diagonal at row " + std::to_string(i));
        }

        forward_levels[i] = level;
    }

    build_level_schedule(forward_levels, ic0.forward_level_ptr, ic0.forward_rows);

    ic0.upper_ptr.assign(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t idx = row_ptr[i]; idx < ic0.diag_pos[i]; ++idx) {
            ic0.upper_ptr[col_idx[idx] + 1]++;
        }
    }
    for (size_t i = 1; i < ic0.upper_ptr.size(); ++i) {
        ic0.upper_ptr[i] += ic0.upper_ptr[i - 1];
    }

    ic0.upper_row_idx.assign(ic0.upper_ptr.back(), 0);
    ic0.upper_pos.assign(ic0.upper_ptr.back(), 0);
    std::vector<size_t> upper_offsets = ic0.upper_ptr;
    for (size_t row = 0; row < n; ++row) {
        for (size_t idx = row_ptr[row]; idx < ic0.diag_pos[row]; ++idx) {
            const size_t col = col_idx[idx];
            const size_t offset = upper_offsets[col]++;
            ic0.upper_row_idx[offset] = row;
            ic0.upper_pos[offset] = idx;
        }
    }

    std::vector<size_t> backward_levels(n, 0);
    for (size_t i = n; i-- > 0;) {
        size_t level = 0;
        for (size_t ptr = ic0.upper_ptr[i]; ptr < ic0.upper_ptr[i + 1]; ++ptr) {
            level = std::max(level, backward_levels[ic0.upper_row_idx[ptr]] + 1);
        }
        backward_levels[i] = level;
    }
    build_level_schedule(backward_levels, ic0.backward_level_ptr, ic0.backward_rows);

    const size_t forward_level_count = ic0.forward_level_ptr.empty() ? 0 : ic0.forward_level_ptr.size() - 1;
    for (size_t level = 0; level < forward_level_count; ++level) {
        const size_t begin = ic0.forward_level_ptr[level];
        const size_t end = ic0.forward_level_ptr[level + 1];
        parallel_for_chunks(begin, end, [&](size_t level_index) {
            const size_t i = ic0.forward_rows[level_index];
            const size_t diag_index = ic0.diag_pos[i];

            double row_sum_abs = 0.0;
            for (size_t idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
                row_sum_abs += std::abs(vals[idx]);
            }

            for (size_t idx = row_ptr[i]; idx < diag_index; ++idx) {
                const size_t j = col_idx[idx];
                double sum = vals[idx];
                size_t pi = row_ptr[i];
                size_t pj = row_ptr[j];

                while (pi < diag_index && pj < ic0.diag_pos[j]) {
                    const size_t ci = col_idx[pi];
                    const size_t cj = col_idx[pj];

                    if (ci >= j || cj >= j) {
                        break;
                    }

                    if (ci == cj) {
                        sum -= ic0.L_vals[pi] * ic0.diag[ci] * ic0.L_vals[pj];
                        ++pi;
                        ++pj;
                    }
                    else if (ci < cj) {
                        ++pi;
                    }
                    else {
                        ++pj;
                    }
                }

                double denom = ic0.diag[j];
                if (!std::isfinite(denom) || std::abs(denom) < absolute_shift) {
                    denom = std::max(absolute_shift, relative_shift * std::max(std::abs(vals[ic0.diag_pos[j]]), 1.0));
                }

                double lij = sum / denom;
                if (!std::isfinite(lij)) {
                    lij = 0.0;
                }
                ic0.L_vals[idx] = lij;
            }

            double diag_value = vals[diag_index];
            for (size_t idx = row_ptr[i]; idx < diag_index; ++idx) {
                const size_t j = col_idx[idx];
                diag_value -= ic0.L_vals[idx] * ic0.L_vals[idx] * ic0.diag[j];
            }
            diag_value += relative_shift * row_sum_abs;

            const double diag_floor = std::max({
                absolute_shift,
                relative_shift * std::max(std::abs(vals[diag_index]), 1.0),
                relative_shift * std::max(row_sum_abs, 1.0)
            });

            if (!std::isfinite(diag_value) || diag_value < diag_floor) {
                diag_value = diag_floor;
            }

            ic0.diag[i] = diag_value;
            ic0.L_vals[diag_index] = 1.0;
        });
    }
}

void applyIC0_preconditioner_host(const IC0Preconditioner& ic0,
                                  const std::vector<double>& b,
                                  std::vector<double>& y,
                                  std::vector<double>& x)
{
    const size_t n = b.size();
    if (y.size() != n) {
        y.resize(n);
    }
    if (x.size() != n) {
        x.resize(n);
    }

    const size_t forward_level_count = ic0.forward_level_ptr.size() - 1;
    for (size_t level = 0; level < forward_level_count; ++level) {
        const size_t begin = ic0.forward_level_ptr[level];
        const size_t end = ic0.forward_level_ptr[level + 1];
        parallel_for_chunks(begin, end, [&](size_t idx) {
            const size_t row = ic0.forward_rows[idx];
            const size_t diag_index = ic0.diag_pos[row];
            double sum = b[row];

            for (size_t k = ic0.row_ptr[row]; k < diag_index; ++k) {
                sum -= ic0.L_vals[k] * y[ic0.col_idx[k]];
            }

            y[row] = std::isfinite(sum) ? sum : 0.0;
        });
    }

    const size_t backward_level_count = ic0.backward_level_ptr.size() - 1;
    for (size_t level = 0; level < backward_level_count; ++level) {
        const size_t begin = ic0.backward_level_ptr[level];
        const size_t end = ic0.backward_level_ptr[level + 1];
        parallel_for_chunks(begin, end, [&](size_t idx) {
            const size_t row = ic0.backward_rows[idx];
            double diag_value = ic0.diag[row];
            if (!std::isfinite(diag_value) || std::abs(diag_value) < 1e-14) {
                diag_value = 1e-14;
            }

            double sum = y[row] / diag_value;
            for (size_t ptr = ic0.upper_ptr[row]; ptr < ic0.upper_ptr[row + 1]; ++ptr) {
                sum -= ic0.L_vals[ic0.upper_pos[ptr]] * x[ic0.upper_row_idx[ptr]];
            }

            x[row] = std::isfinite(sum) ? sum : 0.0;
        });
    }
}

void applyIC0_preconditioner_host(const IC0Preconditioner& ic0,
                                  const std::vector<double>& b,
                                  std::vector<double>& x)
{
    std::vector<double> y;
    applyIC0_preconditioner_host(ic0, b, y, x);
}

void applyIC0_preconditioner(sycl::queue& q,
                             sycl::buffer<size_t>& row_ptr_buf,
                             sycl::buffer<size_t>& col_idx_buf,
                             sycl::buffer<double>& L_buf,
                             sycl::buffer<double>& diag_buf,
                             sycl::buffer<size_t>& diag_pos_buf,
                             const std::vector<size_t>& forward_level_ptr,
                             sycl::buffer<size_t>& forward_rows_buf,
                             const std::vector<size_t>& backward_level_ptr,
                             sycl::buffer<size_t>& backward_rows_buf,
                             sycl::buffer<size_t>& upper_ptr_buf,
                             sycl::buffer<size_t>& upper_row_idx_buf,
                             sycl::buffer<size_t>& upper_pos_buf,
                             sycl::buffer<double>& b_buf,
                             sycl::buffer<double>& y_buf,
                             sycl::buffer<double>& x_buf,
                             size_t n)
{
    const size_t forward_level_count = forward_level_ptr.empty() ? 0 : forward_level_ptr.size() - 1;
    const size_t backward_level_count = backward_level_ptr.empty() ? 0 : backward_level_ptr.size() - 1;
    const size_t avg_forward_width =
        (forward_level_count > 0) ? (n + forward_level_count - 1) / forward_level_count : n;
    const size_t avg_backward_width =
        (backward_level_count > 0) ? (n + backward_level_count - 1) / backward_level_count : n;
    // Keep all GPU computations on device; host path is CPU-only.
    const bool prefer_host_parallel = q.get_device().is_cpu();

    if (prefer_host_parallel) {
        q.wait();

        sycl::host_accessor row_ptr(row_ptr_buf, sycl::read_only);
        sycl::host_accessor col_idx(col_idx_buf, sycl::read_only);
        sycl::host_accessor L(L_buf, sycl::read_only);
        sycl::host_accessor diag(diag_buf, sycl::read_only);
        sycl::host_accessor diag_pos(diag_pos_buf, sycl::read_only);
        sycl::host_accessor forward_rows(forward_rows_buf, sycl::read_only);
        sycl::host_accessor backward_rows(backward_rows_buf, sycl::read_only);
        sycl::host_accessor upper_ptr(upper_ptr_buf, sycl::read_only);
        sycl::host_accessor upper_row_idx(upper_row_idx_buf, sycl::read_only);
        sycl::host_accessor upper_pos(upper_pos_buf, sycl::read_only);
        sycl::host_accessor b(b_buf, sycl::read_only);
        sycl::host_accessor y(y_buf, sycl::read_write);
        sycl::host_accessor x(x_buf, sycl::write_only, sycl::no_init);

        for (size_t level = 0; level < forward_level_count; ++level) {
            const size_t begin = forward_level_ptr[level];
            const size_t end = forward_level_ptr[level + 1];

            parallel_for_chunks(begin, end, [&](size_t idx) {
                const size_t row = forward_rows[idx];
                const size_t diag_index = diag_pos[row];
                double sum = b[row];

                for (size_t k = row_ptr[row]; k < diag_index; ++k) {
                    sum -= L[k] * y[col_idx[k]];
                }

                y[row] = std::isfinite(sum) ? sum : 0.0;
            });
        }

        for (size_t level = 0; level < backward_level_count; ++level) {
            const size_t begin = backward_level_ptr[level];
            const size_t end = backward_level_ptr[level + 1];

            parallel_for_chunks(begin, end, [&](size_t idx) {
                const size_t row = backward_rows[idx];
                double diag_value = diag[row];
                if (!std::isfinite(diag_value) || std::abs(diag_value) < 1e-14) {
                    diag_value = 1e-14;
                }

                double sum = y[row] / diag_value;
                for (size_t ptr = upper_ptr[row]; ptr < upper_ptr[row + 1]; ++ptr) {
                    sum -= L[upper_pos[ptr]] * x[upper_row_idx[ptr]];
                }

                x[row] = std::isfinite(sum) ? sum : 0.0;
            });
        }
        return;
    }

    for (size_t level = 0; level < forward_level_count; ++level) {
        const size_t begin = forward_level_ptr[level];
        const size_t end = forward_level_ptr[level + 1];
        if (end <= begin) {
            continue;
        }

        q.submit([&](sycl::handler& h) {
            auto row_ptr = row_ptr_buf.get_access<sycl::access::mode::read>(h);
            auto col_idx = col_idx_buf.get_access<sycl::access::mode::read>(h);
            auto L = L_buf.get_access<sycl::access::mode::read>(h);
            auto diag_pos = diag_pos_buf.get_access<sycl::access::mode::read>(h);
            auto level_rows = forward_rows_buf.get_access<sycl::access::mode::read>(h);
            auto b = b_buf.get_access<sycl::access::mode::read>(h);
            auto y = y_buf.get_access<sycl::access::mode::read_write>(h);

            h.parallel_for(sycl::range<1>(end - begin), [=](sycl::id<1> idx) {
                const size_t row = level_rows[begin + idx[0]];
                const size_t diag_index = diag_pos[row];
                double sum = b[row];

                for (size_t k = row_ptr[row]; k < diag_index; ++k) {
                    sum -= L[k] * y[col_idx[k]];
                }

                y[row] = std::isfinite(sum) ? sum : 0.0;
            });
        });
    }

    for (size_t level = 0; level < backward_level_count; ++level) {
        const size_t begin = backward_level_ptr[level];
        const size_t end = backward_level_ptr[level + 1];
        if (end <= begin) {
            continue;
        }

        q.submit([&](sycl::handler& h) {
            auto L = L_buf.get_access<sycl::access::mode::read>(h);
            auto diag = diag_buf.get_access<sycl::access::mode::read>(h);
            auto level_rows = backward_rows_buf.get_access<sycl::access::mode::read>(h);
            auto upper_ptr = upper_ptr_buf.get_access<sycl::access::mode::read>(h);
            auto upper_row_idx = upper_row_idx_buf.get_access<sycl::access::mode::read>(h);
            auto upper_pos = upper_pos_buf.get_access<sycl::access::mode::read>(h);
            auto y = y_buf.get_access<sycl::access::mode::read>(h);
            auto x = x_buf.get_access<sycl::access::mode::read_write>(h);

            h.parallel_for(sycl::range<1>(end - begin), [=](sycl::id<1> idx) {
                const size_t row = level_rows[begin + idx[0]];
                double diag_value = diag[row];
                if (!std::isfinite(diag_value) || std::abs(diag_value) < 1e-14) {
                    diag_value = 1e-14;
                }

                double sum = y[row] / diag_value;
                for (size_t ptr = upper_ptr[row]; ptr < upper_ptr[row + 1]; ++ptr) {
                    sum -= L[upper_pos[ptr]] * x[upper_row_idx[ptr]];
                }

                x[row] = std::isfinite(sum) ? sum : 0.0;
            });
        });
    }

    q.wait();
}
