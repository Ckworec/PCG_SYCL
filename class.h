#pragma once

#include <stdio.h>
#include <math.h>
#include <thread>
#include <typeinfo>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sycl/sycl.hpp>

using namespace std;

template<typename T>
class CSR_matrix {
    private:
        vector<size_t> m_row_ptr;
        vector<size_t> m_col_ind;
        vector<T> m_val;
        // Compact 32-bit indices for GPU kernels — built lazily on first use.
        // Halves SpMV column-index bandwidth vs 64-bit size_t.
        mutable vector<uint32_t> m_col_ind_u32;
        mutable vector<uint32_t> m_row_ptr_u32;
        size_t m_rows, m_cols, m_nnz;

    public:
        CSR_matrix(vector<size_t>& row_ptr, vector<size_t>& col_ind, vector<T>& val);
        CSR_matrix();

        ~CSR_matrix();

        void read_mat(const string& name_file);
        void read_mat_matrix_market_symmetric(const string& name_file);
        vector<size_t>& row_ptr_ref();
        vector<size_t>& col_ind_ref();
        vector<T>& val_ref();
        const vector<size_t>& row_ptr_ref() const;
        const vector<size_t>& col_ind_ref() const;
        const vector<T>& val_ref() const;
        vector<uint32_t>& col_ind_u32_ref();
        vector<uint32_t>& row_ptr_u32_ref();
        vector<size_t> take_row_ptr();
        vector<size_t> take_col_ind();
        vector<T> take_val();
        size_t take_rows();
        size_t take_cols();
        size_t take_nnz();
        size_t memory_usage_bytes() const;
};
