#include "func.h"

using namespace std;

void read_vec(vector<double>& vec,
              string& name_file)
{
    ifstream vec_file(name_file);

    if (!vec_file.is_open()) {
        throw runtime_error("Could not open vector file: " + name_file);
    }

    size_t vec_size = 0;
    vec_file >> vec_size;
    vec.resize(vec_size);

    for (size_t i = 0; i < vec_size; ++i) {
        vec_file >> vec[i];
    }
}

void copy_from_nvidia(sycl::queue& q,
                      sycl::buffer<double>& a_buf,
                      double& a)
{
    q.submit([&](sycl::handler& h) {
        sycl::accessor acc(a_buf, h, sycl::read_only);
        h.copy(acc, &a);
    }).wait_and_throw();
}

void copy_from_nvidia(sycl::queue& q,
                      sycl::buffer<double>& a_buf,
                      vector<double>& a)
{
    q.submit([&](sycl::handler& h) {
        sycl::accessor acc(a_buf, h, sycl::read_only);
        h.copy(acc, a.data());
    }).wait_and_throw();
}

double check_result(vector<double> first,
                    vector<double> second)
{
    double average = 0.0;

    for (size_t i = 0; i < first.size(); ++i) {
        average += fabs(first[i] - second[i]);
    }

    return average / first.size();
}

double check_result(vector<double> x,
                    vector<double> b,
                    CSR_matrix<double>& mat)
{
    double b_norm = scalar_product(b, b);

    const vector<size_t>& row_ptr = mat.row_ptr_ref();
    const vector<size_t>& col_ind = mat.col_ind_ref();
    const vector<double>& val = mat.val_ref();
    vector<double> Ax(x.size(), 0.0);

    for (size_t i = 0; i + 1 < row_ptr.size(); ++i) {
        for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            Ax[i] += val[j] * x[col_ind[j]];
        }
    }

    for (size_t i = 0; i < x.size(); ++i) {
        Ax[i] -= b[i];
    }

    const double res = scalar_product(Ax, Ax);
    return sqrt(res) / sqrt(b_norm) * 100.0;
}
