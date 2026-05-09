#include "func.h"

int main(void){
    std::cout << "Enter number file: ";
    std::string number;
    std::cin >> number;

    std::string prefix = "mat_vec/";
    // std::string A_name = prefix + "A" + number + "_bad.txt";
    // std::string B_name = prefix + "B" + number + "_bad.vec";
    std::string A_name = prefix + "A3d" + number + "k.txt";
    std::string B_name = prefix + "B3d" + number + "k.vec";
    std::string report_name = "results.xlsx";
    // std::string A_name = prefix + "A" + number + ".txt";
    // std::string B_name = prefix + "B" + number + ".vec";
    std::string CPU = "CPU", GPU = "GPU";
    int iter_numb = 1;

    std::vector<double> b;
    CSR_matrix<double> mat;

    if (number == "5502") {
        mat.read_mat_matrix_market_symmetric(A_name);
    } else {
        mat.read_mat(A_name);
    }

    read_vec(b, B_name);

    std::vector<double> X1(b.size());
    std::vector<double> X2(b.size());
    std::vector<double> X3(b.size());
    const size_t block_size = 7;
    const size_t polynomial_degree = 4;

    std::cout << "Matrix size: " <<  mat.take_rows() << std::endl << std::endl;

    if (number == "5502") {
        const double gib = 1024.0 * 1024.0 * 1024.0;
        const double host_matrix_gib = static_cast<double>(mat.memory_usage_bytes()) / gib;
        const double host_and_device_matrix_gib = 2.0 * host_matrix_gib;

        std::cout << "Matrix nnz in CSR after symmetric expansion: " << mat.take_nnz() << std::endl;
        std::cout << "Matrix memory in host CSR: " << host_matrix_gib << " GiB" << std::endl;
        std::cout << "Base host+device matrix footprint: " << host_and_device_matrix_gib
                  << " GiB" << std::endl << std::endl;
    }

    auto run_and_report = [&](const std::string& method_name, auto cpu_run, auto gpu_run, auto mkl_run) {
        std::fill(X1.begin(), X1.end(), 0.0);
        std::fill(X2.begin(), X2.end(), 0.0);
        std::fill(X3.begin(), X3.end(), 0.0);

        std::cout << "=== " << method_name << " ===" << std::endl;
        BenchmarkResult cpu_result = cpu_run();
        BenchmarkResult gpu_result = gpu_run();
        BenchmarkResult mkl_result = mkl_run();

        cpu_result.error_pct = check_result(X1, b, mat);
        gpu_result.error_pct = check_result(X2, b, mat);
        mkl_result.error_pct = check_result(X3, b, mat);

        append_benchmark_to_xlsx(report_name, A_name, cpu_result);
        append_benchmark_to_xlsx(report_name, A_name, gpu_result);
        append_benchmark_to_xlsx(report_name, A_name, mkl_result);

        std::cout << "Average error CPU: " << cpu_result.error_pct << "%" << std::endl;
        std::cout << "Average error GPU: " << gpu_result.error_pct << "%" << std::endl;
        std::cout << "Average error MKL: " << mkl_result.error_pct << "%" << std::endl << std::endl;
    };

    run_and_report("CG",
        [&]() { return test_CG_SYCL_plain(mat, X1, b, CPU, iter_numb); },
        [&]() { return test_CG_SYCL_plain(mat, X2, b, GPU, iter_numb); },
        [&]() { return test_CG_MKL_plain(mat, X3, b, iter_numb); });

    run_and_report("Block Jacobi",
        [&]() { return test_CG_SYCL_block_jacobi(mat, X1, b, CPU, iter_numb, block_size); },
        [&]() { return test_CG_SYCL_block_jacobi(mat, X2, b, GPU, iter_numb, block_size); },
        [&]() { return test_CG_MKL_block_jacobi(mat, X3, b, iter_numb, block_size); });

    run_and_report("Jacobi",
        [&]() { return test_CG_SYCL_jacobi(mat, X1, b, CPU, iter_numb); },
        [&]() { return test_CG_SYCL_jacobi(mat, X2, b, GPU, iter_numb); },
        [&]() { return test_CG_MKL_jacobi(mat, X3, b, iter_numb); });

    run_and_report("Chebyshev",
        [&]() { return test_CG_SYCL_chebyshev_adaptive(mat, X1, b, CPU, iter_numb, polynomial_degree); },
        [&]() { return test_CG_SYCL_chebyshev_adaptive(mat, X2, b, GPU, iter_numb, polynomial_degree); },
        [&]() { return test_CG_MKL_chebyshev(mat, X3, b, iter_numb, polynomial_degree); });

    run_and_report("IC0",
        [&]() { return test_CG_SYCL_IC0(mat, X1, b, CPU, iter_numb); },
        [&]() { return test_CG_SYCL_IC0(mat, X2, b, GPU, iter_numb); },
        [&]() { return test_CG_MKL_IC0(mat, X3, b, iter_numb); });

    return 0;
}
