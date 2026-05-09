#include "func.h"
#include <array>
#include <atomic>
#include <cstdint>

using namespace sycl;

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


// =====================================================================
// IC0 — port of the standalone implementation in
// ic0_sycl/ic0_cg_sycl.cpp.
//
//   * ic0_factor                     : level-scheduled diagonal-shifted IC0
//                                      factorisation (rows inside one level
//                                      are independent -> host-parallel).
//   * applyIC0_preconditioner_host   : host apply with two flavours selected
//                                      by env IC0_JACOBI_SWEEPS:
//                                          0      -> exact level-scheduled
//                                                    forward+backward solve
//                                          N >= 1 -> N Jacobi sweeps in each
//                                                    direction (default 2)
//                                      Backed by a persistent worker pool so
//                                      that fork-join cost on the thousands
//                                      of small levels is ~1us per dispatch
//                                      instead of the ~50us std::thread spawn.
//   * applyIC0_preconditioner_jacobi_device :
//                                      GPU SYCL Jacobi-sweep apply. Each
//                                      sweep is one bulk parallel_for over n
//                                      rows; ns total sweeps in each
//                                      direction. Collapses ~1300 per-level
//                                      kernel launches (exact path on GPU)
//                                      down to ~4 bulk launches per apply.
// =====================================================================

namespace ic0_detail {

// Persistent worker pool. Spawning std::threads on Windows costs ~50us,
// which is way too much when IC0 apply has thousands of small levels and
// the CG loop fires thousands of fork-joins. With a pool that lives for
// the entire solve, each dispatch is just a few atomic-store / spin-wait
// hops (~1-2us) and we keep all cores hot through the whole apply.
class WorkerPool {
public:
    explicit WorkerPool(std::size_t nthreads)
        : n_(std::max<std::size_t>(1, nthreads)),
          stop_(false),
          generation_(0),
          done_count_(0),
          fn_ptr_(nullptr),
          begin_(0),
          end_(0)
    {
        threads_.reserve(n_);
        for (std::size_t tid = 0; tid < n_; ++tid) {
            threads_.emplace_back([this, tid]() { worker_loop(tid); });
        }
    }

    ~WorkerPool()
    {
        stop_.store(true, std::memory_order_release);
        generation_.fetch_add(1, std::memory_order_release);
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

    std::size_t num_threads() const { return n_; }

    template <typename Fn>
    void run_for(std::size_t begin, std::size_t end, Fn&& fn)
    {
        if (begin >= end) return;
        const std::size_t work = end - begin;
        if (work < parallel_threshold_) {
            for (std::size_t i = begin; i < end; ++i) fn(i);
            return;
        }
        TrampolineFn trampoline = +[](void* user, std::size_t i) {
            (*static_cast<Fn*>(user))(i);
        };
        Fn local_fn = std::forward<Fn>(fn);
        fn_ptr_      = trampoline;
        fn_user_     = &local_fn;
        begin_       = begin;
        end_         = end;
        done_count_.store(0, std::memory_order_release);
        generation_.fetch_add(1, std::memory_order_acq_rel);
        while (done_count_.load(std::memory_order_acquire) != n_) {
            std::this_thread::yield();
        }
    }

    void set_min_parallel_chunk(std::size_t v) { parallel_threshold_ = v; }

private:
    using TrampolineFn = void(*)(void*, std::size_t);

    void worker_loop(std::size_t tid)
    {
        std::uint64_t local_gen = 0;
        while (true) {
            while (generation_.load(std::memory_order_acquire) == local_gen) {
                std::this_thread::yield();
            }
            if (stop_.load(std::memory_order_acquire)) return;
            local_gen = generation_.load(std::memory_order_acquire);

            const std::size_t work  = end_ - begin_;
            const std::size_t chunk = (work + n_ - 1) / n_;
            const std::size_t lo    = begin_ + tid * chunk;
            const std::size_t hi    = std::min(end_, lo + chunk);
            if (lo < hi) {
                TrampolineFn fn = fn_ptr_;
                void*       u   = fn_user_;
                for (std::size_t i = lo; i < hi; ++i) fn(u, i);
            }
            done_count_.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    std::size_t                       n_;
    std::vector<std::thread>          threads_;
    std::atomic<bool>                 stop_;
    std::atomic<std::uint64_t>        generation_;
    std::atomic<std::size_t>          done_count_;
    TrampolineFn                      fn_ptr_;
    void*                             fn_user_  = nullptr;
    std::size_t                       begin_;
    std::size_t                       end_;
    std::size_t                       parallel_threshold_ = 256;
};

inline WorkerPool& worker_pool()
{
    static WorkerPool pool(std::max(1u, std::thread::hardware_concurrency()));
    return pool;
}

// IC0 produces hundreds of short forward/backward levels (often 1-256 rows
// each). Creating a fresh std::thread on Windows costs ~50us, so spawning a
// thread pool per level burns more time than the actual work. Stay serial
// unless `work` is large enough to amortise the spawn.
template <typename Func>
inline void parallel_for_chunks(std::size_t begin, std::size_t end, Func&& fn,
                                std::size_t min_chunk = 4096)
{
    if (end <= begin) return;

    const std::size_t  work = end - begin;
    if (work < min_chunk) {
        for (std::size_t idx = begin; idx < end; ++idx) fn(idx);
        return;
    }

    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t  thread_count =
        std::min<std::size_t>(hw_threads, (work + min_chunk - 1) / min_chunk);

    if (thread_count <= 1) {
        for (std::size_t idx = begin; idx < end; ++idx) fn(idx);
        return;
    }

    const std::size_t chunk = (work + thread_count - 1) / thread_count;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::size_t t = 0; t < thread_count; ++t) {
        const std::size_t lo = begin + t * chunk;
        const std::size_t hi = std::min(end, lo + chunk);
        if (lo >= hi) continue;
        workers.emplace_back([lo, hi, &fn]() {
            for (std::size_t idx = lo; idx < hi; ++idx) fn(idx);
        });
    }
    for (auto& w : workers) w.join();
}

inline void build_level_schedule(const std::vector<std::size_t>& levels,
                                 std::vector<std::size_t>&       level_ptr,
                                 std::vector<std::size_t>&       rows)
{
    const std::size_t n = levels.size();
    std::size_t max_level = 0;
    for (std::size_t l : levels) max_level = std::max(max_level, l);

    level_ptr.assign(max_level + 2, 0);
    for (std::size_t l : levels) level_ptr[l + 1]++;
    for (std::size_t i = 1; i < level_ptr.size(); ++i) {
        level_ptr[i] += level_ptr[i - 1];
    }
    rows.assign(n, 0);
    auto offsets = level_ptr;
    for (std::size_t r = 0; r < n; ++r) {
        rows[offsets[levels[r]]++] = r;
    }
}

inline int read_jacobi_sweeps_env(int default_sweeps = 2)
{
    if (const char* env = std::getenv("IC0_JACOBI_SWEEPS")) {
        const int v = std::atoi(env);
        if (v >= 0 && v <= 10) return v;
    }
    return default_sweeps;
}

} // namespace ic0_detail

// Diagonal-shifted IC0 factorisation. Numeric phase is parallelised over the
// rows of each forward level (rows within one level are independent of each
// other; only previously-computed levels are read).
void ic0_factor(size_t n,
                const std::vector<size_t>& row_ptr,
                const std::vector<size_t>& col_idx,
                const std::vector<double>& vals,
                IC0Preconditioner& ic0)
{
    using ic0_detail::parallel_for_chunks;
    using ic0_detail::build_level_schedule;

    const size_t not_found = static_cast<size_t>(-1);
    const double rel_shift = 1e-1;
    const double abs_shift = 1e-14;

    // Keep a copy of the matrix structure inside the preconditioner so that
    // host applies (e.g. the MKL CG callback) are self-contained.
    ic0.row_ptr = row_ptr;
    ic0.col_idx = col_idx;

    ic0.L_vals.assign(vals.size(), 0.0);
    ic0.diag.assign(n, 0.0);
    ic0.diag_pos.assign(n, not_found);

    std::vector<size_t> forward_levels(n, 0);
    for (size_t i = 0; i < n; ++i) {
        size_t level = 0;
        for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            const size_t col = col_idx[k];
            if (col == i) {
                ic0.diag_pos[i] = k;
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
            throw std::runtime_error("IC0 needs positive diagonal at row " + std::to_string(i));
        }
        forward_levels[i] = level;
    }
    build_level_schedule(forward_levels, ic0.forward_level_ptr, ic0.forward_rows);

    ic0.upper_ptr.assign(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = row_ptr[i]; k < ic0.diag_pos[i]; ++k) {
            ic0.upper_ptr[col_idx[k] + 1]++;
        }
    }
    for (size_t i = 1; i < ic0.upper_ptr.size(); ++i) {
        ic0.upper_ptr[i] += ic0.upper_ptr[i - 1];
    }

    ic0.upper_row_idx.assign(ic0.upper_ptr.back(), 0);
    ic0.upper_pos.assign(ic0.upper_ptr.back(), 0);
    {
        auto offsets = ic0.upper_ptr;
        for (size_t row = 0; row < n; ++row) {
            for (size_t k = row_ptr[row]; k < ic0.diag_pos[row]; ++k) {
                const size_t col = col_idx[k];
                const size_t off = offsets[col]++;
                ic0.upper_row_idx[off] = row;
                ic0.upper_pos[off]     = k;
            }
        }
    }

    std::vector<size_t> backward_levels(n, 0);
    for (size_t i = n; i-- > 0;) {
        size_t level = 0;
        for (size_t p = ic0.upper_ptr[i]; p < ic0.upper_ptr[i + 1]; ++p) {
            level = std::max(level, backward_levels[ic0.upper_row_idx[p]] + 1);
        }
        backward_levels[i] = level;
    }
    build_level_schedule(backward_levels, ic0.backward_level_ptr, ic0.backward_rows);

    const size_t flcount =
        ic0.forward_level_ptr.empty() ? 0 : ic0.forward_level_ptr.size() - 1;
    for (size_t level = 0; level < flcount; ++level) {
        const size_t lb = ic0.forward_level_ptr[level];
        const size_t le = ic0.forward_level_ptr[level + 1];

        parallel_for_chunks(lb, le, [&](size_t li) {
            const size_t i  = ic0.forward_rows[li];
            const size_t di = ic0.diag_pos[i];

            double row_sum_abs = 0.0;
            for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                row_sum_abs += std::abs(vals[k]);
            }

            for (size_t idx = row_ptr[i]; idx < di; ++idx) {
                const size_t j = col_idx[idx];
                double sum  = vals[idx];
                size_t pi = row_ptr[i];
                size_t pj = row_ptr[j];

                while (pi < di && pj < ic0.diag_pos[j]) {
                    const size_t ci = col_idx[pi];
                    const size_t cj = col_idx[pj];
                    if (ci >= j || cj >= j) break;
                    if (ci == cj) {
                        sum -= ic0.L_vals[pi] * ic0.diag[ci] * ic0.L_vals[pj];
                        ++pi;
                        ++pj;
                    } else if (ci < cj) {
                        ++pi;
                    } else {
                        ++pj;
                    }
                }

                double denom = ic0.diag[j];
                if (!std::isfinite(denom) || std::abs(denom) < abs_shift) {
                    denom = std::max(abs_shift,
                                     rel_shift * std::max(std::abs(vals[ic0.diag_pos[j]]), 1.0));
                }
                double lij = sum / denom;
                if (!std::isfinite(lij)) lij = 0.0;
                ic0.L_vals[idx] = lij;
            }

            double diag_value = vals[di];
            for (size_t idx = row_ptr[i]; idx < di; ++idx) {
                const size_t j = col_idx[idx];
                diag_value -= ic0.L_vals[idx] * ic0.L_vals[idx] * ic0.diag[j];
            }
            diag_value += rel_shift * row_sum_abs;

            const double diag_floor = std::max({
                abs_shift,
                rel_shift * std::max(std::abs(vals[di]), 1.0),
                rel_shift * std::max(row_sum_abs, 1.0)
            });
            if (!std::isfinite(diag_value) || diag_value < diag_floor) {
                diag_value = diag_floor;
            }
            ic0.diag[i]    = diag_value;
            ic0.L_vals[di] = 1.0;
        });
    }
}

namespace ic0_detail {

// Exact (level-scheduled) apply. Forward+backward triangular solve along
// the IC0 dependency DAG, using the persistent WorkerPool.
inline void apply_host_exact(const IC0Preconditioner&   ic0,
                             const std::vector<double>& src,
                             std::vector<double>&       y,
                             std::vector<double>&       dst)
{
    const std::size_t flc =
        ic0.forward_level_ptr.empty() ? 0 : ic0.forward_level_ptr.size() - 1;
    const std::size_t blc =
        ic0.backward_level_ptr.empty() ? 0 : ic0.backward_level_ptr.size() - 1;
    auto& pool = worker_pool();

    for (std::size_t level = 0; level < flc; ++level) {
        const std::size_t lb = ic0.forward_level_ptr[level];
        const std::size_t le = ic0.forward_level_ptr[level + 1];
        pool.run_for(lb, le, [&](std::size_t idx) {
            const std::size_t row = ic0.forward_rows[idx];
            const std::size_t di  = ic0.diag_pos[row];
            double sum = src[row];
            for (std::size_t k = ic0.row_ptr[row]; k < di; ++k) {
                sum -= ic0.L_vals[k] * y[ic0.col_idx[k]];
            }
            y[row] = std::isfinite(sum) ? sum : 0.0;
        });
    }
    for (std::size_t level = 0; level < blc; ++level) {
        const std::size_t lb = ic0.backward_level_ptr[level];
        const std::size_t le = ic0.backward_level_ptr[level + 1];
        pool.run_for(lb, le, [&](std::size_t idx) {
            const std::size_t row = ic0.backward_rows[idx];
            double d = ic0.diag[row];
            if (!std::isfinite(d) || std::abs(d) < 1e-14) d = 1e-14;
            double sum = y[row] / d;
            for (std::size_t pp = ic0.upper_ptr[row]; pp < ic0.upper_ptr[row + 1]; ++pp) {
                sum -= ic0.L_vals[ic0.upper_pos[pp]] * dst[ic0.upper_row_idx[pp]];
            }
            dst[row] = std::isfinite(sum) ? sum : 0.0;
        });
    }
}

// Jacobi-sweep apply. Mirrors the GPU implementation: each sweep is one bulk
// parallel pass over n rows, no per-level dispatch. ns total sweeps per
// direction (1 init "y_curr = src" + ns-1 real). Approximates L^{-1} well
// after 2-3 sweeps for diagonally dominant 3D matrices, at a fraction of the
// dispatch cost on big matrices.
inline void apply_host_jacobi(const IC0Preconditioner&   ic0,
                              const std::vector<double>& src,
                              std::vector<double>&       y,
                              std::vector<double>&       y_tmp,
                              std::vector<double>&       dst,
                              int                        ns)
{
    const std::size_t n = src.size();
    auto& pool = worker_pool();

    auto fwd_sweep = [&](const std::vector<double>& yin,
                         std::vector<double>&       yout,
                         bool                       init) {
        pool.run_for(0, n, [&, init](std::size_t i) {
            double s = src[i];
            if (!init) {
                const std::size_t di = ic0.diag_pos[i];
                for (std::size_t k = ic0.row_ptr[i]; k < di; ++k) {
                    s -= ic0.L_vals[k] * yin[ic0.col_idx[k]];
                }
            }
            yout[i] = std::isfinite(s) ? s : 0.0;
        });
    };

    auto bwd_sweep = [&](const std::vector<double>& yfwd,
                         const std::vector<double>& xin,
                         std::vector<double>&       xout,
                         bool                       init) {
        pool.run_for(0, n, [&, init](std::size_t i) {
            double d = ic0.diag[i];
            if (!std::isfinite(d) || std::abs(d) < 1e-14) d = 1e-14;
            double s = yfwd[i] / d;
            if (!init) {
                const std::size_t s0 = ic0.upper_ptr[i];
                const std::size_t s1 = ic0.upper_ptr[i + 1];
                for (std::size_t pp = s0; pp < s1; ++pp) {
                    s -= ic0.L_vals[ic0.upper_pos[pp]] * xin[ic0.upper_row_idx[pp]];
                }
            }
            xout[i] = std::isfinite(s) ? s : 0.0;
        });
    };

    // Pick parity so the LAST forward sweep writes into y (the backward
    // phase reads y). Same trick as the GPU implementation.
    const bool fwd_last_into_y = (ns % 2 == 1);
    std::vector<double>* fwd_first  = fwd_last_into_y ? &y     : &y_tmp;
    std::vector<double>* fwd_second = fwd_last_into_y ? &y_tmp : &y;

    fwd_sweep(*fwd_first, *fwd_first, /*init=*/true);
    for (int s = 1; s < ns; ++s) {
        std::vector<double>* yin  = (s % 2 == 1) ? fwd_first  : fwd_second;
        std::vector<double>* yout = (s % 2 == 1) ? fwd_second : fwd_first;
        fwd_sweep(*yin, *yout, /*init=*/false);
    }
    // Forward result lives in y.

    // Backward: result must end in dst.
    const bool bwd_starts_in_dst = (ns % 2 == 1);
    std::vector<double>* bwd_first  = bwd_starts_in_dst ? &dst   : &y_tmp;
    std::vector<double>* bwd_second = bwd_starts_in_dst ? &y_tmp : &dst;

    bwd_sweep(y, *bwd_first, *bwd_first, /*init=*/true);
    for (int s = 1; s < ns; ++s) {
        std::vector<double>* xin  = (s % 2 == 1) ? bwd_first  : bwd_second;
        std::vector<double>* xout = (s % 2 == 1) ? bwd_second : bwd_first;
        bwd_sweep(y, *xin, *xout, /*init=*/false);
    }
    // Backward result in dst.
}

} // namespace ic0_detail

void applyIC0_preconditioner_host(const IC0Preconditioner&   ic0,
                                  const std::vector<double>& src,
                                  std::vector<double>&       y,
                                  std::vector<double>&       dst)
{
    const std::size_t n = src.size();
    if (y.size()   != n) y.assign(n, 0.0);
    if (dst.size() != n) dst.assign(n, 0.0);

    const int ns = ic0_detail::read_jacobi_sweeps_env(2);
    if (ns <= 0) {
        ic0_detail::apply_host_exact(ic0, src, y, dst);
    } else {
        thread_local std::vector<double> y_tmp;
        if (y_tmp.size() != n) y_tmp.assign(n, 0.0);
        ic0_detail::apply_host_jacobi(ic0, src, y, y_tmp, dst, ns);
    }
}

// SYCL Jacobi-sweep IC0 apply for the GPU. Each sweep is a single bulk
// parallel_for over n rows; ns total sweeps in each direction. y_buf and
// y2_buf alternate as ping-pong scratch; final result lands in dst_buf.
void applyIC0_preconditioner_jacobi_device(sycl::queue&              q,
                                           sycl::buffer<size_t>&     row_ptr_buf,
                                           sycl::buffer<size_t>&     col_idx_buf,
                                           sycl::buffer<double>&     L_buf,
                                           sycl::buffer<double>&     diag_buf,
                                           sycl::buffer<size_t>&     diag_pos_buf,
                                           sycl::buffer<size_t>&     upper_ptr_buf,
                                           sycl::buffer<size_t>&     upper_row_idx_buf,
                                           sycl::buffer<size_t>&     upper_pos_buf,
                                           sycl::buffer<double>&     src_buf,
                                           sycl::buffer<double>&     y_buf,
                                           sycl::buffer<double>&     y2_buf,
                                           sycl::buffer<double>&     dst_buf,
                                           size_t                    n,
                                           int                       ns)
{
    if (ns < 1) ns = 1;

    auto fwd_sweep = [&](sycl::buffer<double>& src_in,
                         sycl::buffer<double>& y_in,
                         sycl::buffer<double>& y_out,
                         bool                  init) {
        q.submit([&, init](sycl::handler& h) {
            sycl::accessor rp_acc  (row_ptr_buf,  h, sycl::read_only);
            sycl::accessor ci_acc  (col_idx_buf,  h, sycl::read_only);
            sycl::accessor L_acc   (L_buf,        h, sycl::read_only);
            sycl::accessor dp_acc  (diag_pos_buf, h, sycl::read_only);
            sycl::accessor src_acc (src_in,       h, sycl::read_only);
            sycl::accessor yin_acc (y_in,         h, sycl::read_only);
            sycl::accessor yout_acc(y_out,        h, sycl::write_only, sycl::no_init);
            const bool ini = init;
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                double s = src_acc[i];
                if (!ini) {
                    const size_t di = dp_acc[i];
                    for (size_t k = rp_acc[i]; k < di; ++k) {
                        s -= L_acc[k] * yin_acc[ci_acc[k]];
                    }
                }
                yout_acc[i] = std::isfinite(s) ? s : 0.0;
            });
        });
    };

    auto bwd_sweep = [&](sycl::buffer<double>& y_fwd,
                         sycl::buffer<double>& x_in,
                         sycl::buffer<double>& x_out,
                         bool                  init) {
        q.submit([&, init](sycl::handler& h) {
            sycl::accessor L_acc    (L_buf,             h, sycl::read_only);
            sycl::accessor dg_acc   (diag_buf,          h, sycl::read_only);
            sycl::accessor up_acc   (upper_ptr_buf,     h, sycl::read_only);
            sycl::accessor uri_acc  (upper_row_idx_buf, h, sycl::read_only);
            sycl::accessor upos_acc (upper_pos_buf,     h, sycl::read_only);
            sycl::accessor y_acc    (y_fwd,             h, sycl::read_only);
            sycl::accessor xin_acc  (x_in,              h, sycl::read_only);
            sycl::accessor xout_acc (x_out,             h, sycl::write_only, sycl::no_init);
            const bool ini = init;
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                double d = dg_acc[i];
                if (!std::isfinite(d) || std::abs(d) < 1e-14) d = 1e-14;
                double s = y_acc[i] / d;
                if (!ini) {
                    const size_t s0 = up_acc[i];
                    const size_t s1 = up_acc[i + 1];
                    for (size_t pp = s0; pp < s1; ++pp) {
                        s -= L_acc[upos_acc[pp]] * xin_acc[uri_acc[pp]];
                    }
                }
                xout_acc[i] = std::isfinite(s) ? s : 0.0;
            });
        });
    };

    // Forward: parity so that the LAST sweep writes into y_buf.
    const bool fwd_last_into_y_buf = (ns % 2 == 1);
    sycl::buffer<double>* fwd_first  = fwd_last_into_y_buf ? &y_buf  : &y2_buf;
    sycl::buffer<double>* fwd_second = fwd_last_into_y_buf ? &y2_buf : &y_buf;

    fwd_sweep(src_buf, *fwd_first, *fwd_first, /*init=*/true);
    for (int s = 1; s < ns; ++s) {
        sycl::buffer<double>* yin  = (s % 2 == 1) ? fwd_first  : fwd_second;
        sycl::buffer<double>* yout = (s % 2 == 1) ? fwd_second : fwd_first;
        fwd_sweep(src_buf, *yin, *yout, /*init=*/false);
    }
    // Forward result lives in y_buf.

    // Backward: result must end in dst_buf.
    const bool bwd_starts_in_dst = (ns % 2 == 1);
    sycl::buffer<double>* bwd_first  = bwd_starts_in_dst ? &dst_buf : &y2_buf;
    sycl::buffer<double>* bwd_second = bwd_starts_in_dst ? &y2_buf  : &dst_buf;

    bwd_sweep(y_buf, *bwd_first, *bwd_first, /*init=*/true);
    for (int s = 1; s < ns; ++s) {
        sycl::buffer<double>* xin  = (s % 2 == 1) ? bwd_first  : bwd_second;
        sycl::buffer<double>* xout = (s % 2 == 1) ? bwd_second : bwd_first;
        bwd_sweep(y_buf, *xin, *xout, /*init=*/false);
    }
    // Result now in dst_buf.
}
