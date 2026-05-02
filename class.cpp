#include "class.h"
#include <sstream>

namespace {

std::string read_next_matrix_market_line(std::ifstream& file)
{
    std::string line;
    while (std::getline(file, line)) {
        const size_t first_non_space = line.find_first_not_of(" \t\r\n");
        if (first_non_space == std::string::npos) {
            continue;
        }

        if (line[first_non_space] == '%') {
            continue;
        }

        return line.substr(first_non_space);
    }

    return {};
}

template <typename T>
void sort_csr_rows(std::vector<size_t>& row_ptr,
                   std::vector<size_t>& col_ind,
                   std::vector<T>& val)
{
    const size_t n = row_ptr.size() > 0 ? row_ptr.size() - 1 : 0;

    for (size_t i = 0; i < n; ++i) {
        const size_t start = row_ptr[i];
        const size_t end = row_ptr[i + 1];
        const size_t len = end - start;

        std::vector<std::pair<size_t, T>> row;
        row.reserve(len);

        for (size_t k = start; k < end; ++k) {
            row.emplace_back(col_ind[k], val[k]);
        }

        std::sort(row.begin(), row.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        for (size_t k = 0; k < len; ++k) {
            col_ind[start + k] = row[k].first;
            val[start + k] = row[k].second;
        }
    }
}

} // namespace

template<typename T>
CSR_matrix<T>::CSR_matrix() {
    m_row_ptr.push_back(0);
    m_row_ptr.push_back(1);
    m_col_ind.push_back(0);
    m_val.push_back(1);
    m_rows = 1;
    m_cols = 1;
    m_nnz = 1;
}

template<typename T>
CSR_matrix<T>::CSR_matrix(vector<size_t>& row_ptr, vector<size_t>& col_ind, vector<T>& val) {
    m_row_ptr = row_ptr;
    m_col_ind = col_ind;
    m_val = val;
    m_rows = row_ptr.empty() ? 0 : row_ptr.size() - 1;
    m_cols = row_ptr.empty() ? 0 : row_ptr.size() - 1;
    m_nnz = val.size();
}

template<typename T>
void CSR_matrix<T>::read_mat(const string& name_file) {
    std::ifstream mat_file(name_file);

    if (!mat_file.is_open()) {
        throw std::runtime_error("Could not open matrix file: " + name_file);
    }

    std::ostringstream filtered_content;
    std::string line;

    while (std::getline(mat_file, line)) {
        const size_t first_non_space = line.find_first_not_of(" \t\r\n");

        if (first_non_space == std::string::npos) {
            continue;
        }

        if (line[first_non_space] == '%') {
            continue;
        }

        filtered_content << line << '\n';
    }

    std::istringstream parser(filtered_content.str());

    if (!(parser >> m_rows >> m_nnz)) {
        throw std::runtime_error("Failed to parse matrix header: " + name_file);
    }

    m_cols = m_rows;

    m_row_ptr.resize(m_rows + 1);
    m_col_ind.resize(m_nnz);
    m_val.resize(m_nnz);

    for (size_t i = 0; i <= m_rows; ++i) {
        if (!(parser >> m_row_ptr[i])) {
            throw std::runtime_error("Failed to parse matrix row_ptr: " + name_file);
        }
    }

    for (size_t i = 0; i < m_nnz; ++i) {
        if (!(parser >> m_col_ind[i])) {
            throw std::runtime_error("Failed to parse matrix col_ind: " + name_file);
        }
    }
    
    for (size_t i = 0; i < m_nnz; ++i) {
        if (!(parser >> m_val[i])) {
            throw std::runtime_error("Failed to parse matrix values: " + name_file);
        }
    }

    mat_file.close();

    sort_csr_rows(m_row_ptr, m_col_ind, m_val);
}

template<typename T>
void CSR_matrix<T>::read_mat_matrix_market_symmetric(const string& name_file) {
    std::ifstream mat_file(name_file);

    if (!mat_file.is_open()) {
        throw std::runtime_error("Could not open matrix file: " + name_file);
    }

    std::string banner;
    if (!std::getline(mat_file, banner)) {
        throw std::runtime_error("Failed to read MatrixMarket banner: " + name_file);
    }

    const std::string expected_banner = "%%MatrixMarket matrix coordinate real symmetric";
    if (banner.find(expected_banner) != 0) {
        throw std::runtime_error("Unsupported MatrixMarket format: " + name_file);
    }

    const std::string dims_line = read_next_matrix_market_line(mat_file);
    if (dims_line.empty()) {
        throw std::runtime_error("Failed to read MatrixMarket dimensions: " + name_file);
    }

    size_t rows = 0;
    size_t cols = 0;
    size_t input_nnz = 0;
    {
        std::istringstream dims_parser(dims_line);
        if (!(dims_parser >> rows >> cols >> input_nnz)) {
            throw std::runtime_error("Failed to parse MatrixMarket dimensions: " + name_file);
        }
    }

    if (rows != cols) {
        throw std::runtime_error("Only square symmetric MatrixMarket matrices are supported: " + name_file);
    }

    m_rows = rows;
    m_cols = cols;
    m_row_ptr.assign(m_rows + 1, 0);

    size_t expanded_nnz = 0;
    std::string line;

    while (std::getline(mat_file, line)) {
        const size_t first_non_space = line.find_first_not_of(" \t\r\n");
        if (first_non_space == std::string::npos || line[first_non_space] == '%') {
            continue;
        }

        std::istringstream entry_parser(line.substr(first_non_space));
        size_t row = 0;
        size_t col = 0;
        T value = 0;

        if (!(entry_parser >> row >> col >> value)) {
            throw std::runtime_error("Failed to parse MatrixMarket entry: " + name_file);
        }

        if (row == 0 || col == 0 || row > m_rows || col > m_cols) {
            throw std::runtime_error("MatrixMarket entry index is out of range: " + name_file);
        }

        --row;
        --col;

        ++m_row_ptr[row + 1];
        ++expanded_nnz;

        if (row != col) {
            ++m_row_ptr[col + 1];
            ++expanded_nnz;
        }
    }

    for (size_t i = 1; i <= m_rows; ++i) {
        m_row_ptr[i] += m_row_ptr[i - 1];
    }

    m_nnz = expanded_nnz;
    m_col_ind.resize(m_nnz);
    m_val.resize(m_nnz);

    std::vector<size_t> next_pos = m_row_ptr;

    mat_file.close();
    mat_file.open(name_file);
    if (!mat_file.is_open()) {
        throw std::runtime_error("Failed to reopen MatrixMarket file: " + name_file);
    }

    if (!std::getline(mat_file, banner)) {
        throw std::runtime_error("Failed to reread MatrixMarket banner: " + name_file);
    }

    if (read_next_matrix_market_line(mat_file).empty()) {
        throw std::runtime_error("Failed to reread MatrixMarket dimensions: " + name_file);
    }

    size_t parsed_entries = 0;

    while (std::getline(mat_file, line)) {
        const size_t first_non_space = line.find_first_not_of(" \t\r\n");
        if (first_non_space == std::string::npos || line[first_non_space] == '%') {
            continue;
        }

        std::istringstream entry_parser(line.substr(first_non_space));
        size_t row = 0;
        size_t col = 0;
        T value = 0;

        if (!(entry_parser >> row >> col >> value)) {
            throw std::runtime_error("Failed to parse MatrixMarket entry during fill: " + name_file);
        }

        --row;
        --col;

        const size_t pos = next_pos[row]++;
        m_col_ind[pos] = col;
        m_val[pos] = value;

        if (row != col) {
            const size_t sym_pos = next_pos[col]++;
            m_col_ind[sym_pos] = row;
            m_val[sym_pos] = value;
        }

        ++parsed_entries;
    }

    if (parsed_entries != input_nnz) {
        throw std::runtime_error("MatrixMarket entry count does not match header: " + name_file);
    }

    sort_csr_rows(m_row_ptr, m_col_ind, m_val);
}

template<typename T>
CSR_matrix<T>::~CSR_matrix() {
    m_row_ptr.clear();
    m_col_ind.clear();
    m_val.clear();
}

template<typename T>
vector<size_t>& CSR_matrix<T>::row_ptr_ref() {
    return m_row_ptr;
}

template<typename T>
vector<size_t>& CSR_matrix<T>::col_ind_ref() {
    return m_col_ind;
}

template<typename T>
vector<T>& CSR_matrix<T>::val_ref() {
    return m_val;
}

template<typename T>
const vector<size_t>& CSR_matrix<T>::row_ptr_ref() const {
    return m_row_ptr;
}

template<typename T>
const vector<size_t>& CSR_matrix<T>::col_ind_ref() const {
    return m_col_ind;
}

template<typename T>
const vector<T>& CSR_matrix<T>::val_ref() const {
    return m_val;
}

template<typename T>
vector<size_t> CSR_matrix<T>::take_row_ptr() {
    return m_row_ptr;
}

template<typename T>
vector<size_t> CSR_matrix<T>::take_col_ind() {
    return m_col_ind;
}

template<typename T>
vector<T> CSR_matrix<T>::take_val() {
    return m_val;
}

template<typename T>
size_t CSR_matrix<T>::take_rows() {
    return m_rows;
}

template<typename T>
size_t CSR_matrix<T>::take_cols() {
    return m_cols;
}

template<typename T>
size_t CSR_matrix<T>::take_nnz() {
    return m_nnz;
}

template<typename T>
size_t CSR_matrix<T>::memory_usage_bytes() const {
    return m_row_ptr.size() * sizeof(size_t)
        + m_col_ind.size() * sizeof(size_t)
        + m_val.size() * sizeof(T);
}

template<typename T>
vector<uint32_t>& CSR_matrix<T>::col_ind_u32_ref() {
    if (m_col_ind_u32.size() != m_col_ind.size()) {
        m_col_ind_u32.resize(m_col_ind.size());
        const size_t n = m_col_ind.size();
        for (size_t k = 0; k < n; ++k) {
            m_col_ind_u32[k] = static_cast<uint32_t>(m_col_ind[k]);
        }
    }
    return m_col_ind_u32;
}

template<typename T>
vector<uint32_t>& CSR_matrix<T>::row_ptr_u32_ref() {
    if (m_row_ptr_u32.size() != m_row_ptr.size()) {
        m_row_ptr_u32.resize(m_row_ptr.size());
        const size_t n = m_row_ptr.size();
        for (size_t k = 0; k < n; ++k) {
            m_row_ptr_u32[k] = static_cast<uint32_t>(m_row_ptr[k]);
        }
    }
    return m_row_ptr_u32;
}

template class CSR_matrix<double>;
