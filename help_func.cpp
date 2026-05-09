#include "func.h"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>

using namespace std;
namespace fs = std::filesystem;

static string format_double_as_text(double value)
{
    ostringstream out;
    out << setprecision(15) << value;
    return out.str();
}

static string build_xlsx_source_path(const string& xlsx_path)
{
    return xlsx_path + ".data";
}

static string escape_xml(const string& value)
{
    string escaped;
    escaped.reserve(value.size());

    for (char ch : value) {
        switch (ch) {
        case '&':
            escaped += "&amp;";
            break;
        case '<':
            escaped += "&lt;";
            break;
        case '>':
            escaped += "&gt;";
            break;
        case '"':
            escaped += "&quot;";
            break;
        case '\'':
            escaped += "&apos;";
            break;
        default:
            escaped += ch;
            break;
        }
    }

    return escaped;
}

static string escape_powershell_single_quoted(const string& value)
{
    string escaped;
    escaped.reserve(value.size());

    for (char ch : value) {
        if (ch == '\'') {
            escaped += "''";
        }
        else {
            escaped += ch;
        }
    }

    return escaped;
}

static vector<string> split_semicolon_line(const string& line)
{
    vector<string> fields;
    string field;
    stringstream stream(line);

    while (getline(stream, field, ';')) {
        fields.push_back(field);
    }

    if (!line.empty() && line.back() == ';') {
        fields.emplace_back();
    }

    return fields;
}

static vector<vector<string>> read_semicolon_table(const string& source_path)
{
    vector<vector<string>> rows;
    ifstream input(source_path);
    string line;

    while (getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            rows.push_back(split_semicolon_line(line));
        }
    }

    return rows;
}

static string excel_column_name(size_t index)
{
    string name;
    size_t current = index;

    do {
        name.insert(name.begin(), static_cast<char>('A' + (current % 26)));
        current = current / 26;
        if (current == 0) {
            break;
        }
        --current;
    } while (true);

    return name;
}

static void write_text_file(const fs::path& path, const string& content)
{
    ofstream output(path, ios::binary);
    output << content;
}

static bool rebuild_xlsx_from_source(const string& xlsx_path)
{
    const string source_path = build_xlsx_source_path(xlsx_path);
    const vector<vector<string>> rows = read_semicolon_table(source_path);

    if (rows.empty()) {
        return false;
    }

    size_t max_columns = 0;
    for (const auto& row : rows) {
        max_columns = max(max_columns, row.size());
    }

    vector<size_t> widths(max_columns, 0);
    for (const auto& row : rows) {
        for (size_t col = 0; col < row.size(); ++col) {
            widths[col] = max(widths[col], row[col].size());
        }
    }

    fs::path xlsx_file = fs::absolute(fs::path(xlsx_path));
    fs::path temp_dir = xlsx_file;
    temp_dir += ".tmpdir";

    std::error_code ec;
    fs::remove_all(temp_dir, ec);
    fs::create_directories(temp_dir / "_rels");
    fs::create_directories(temp_dir / "xl" / "_rels");
    fs::create_directories(temp_dir / "xl" / "worksheets");

    write_text_file(temp_dir / "[Content_Types].xml",
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>"
        "<Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
        "<Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>"
        "</Types>");

    write_text_file(temp_dir / "_rels" / ".rels",
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>"
        "</Relationships>");

    write_text_file(temp_dir / "xl" / "workbook.xml",
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
        "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        "<sheets><sheet name=\"Benchmarks\" sheetId=\"1\" r:id=\"rId1\"/></sheets>"
        "</workbook>");

    write_text_file(temp_dir / "xl" / "_rels" / "workbook.xml.rels",
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>"
        "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>"
        "</Relationships>");

    write_text_file(temp_dir / "xl" / "styles.xml",
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
        "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
        "<fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/><family val=\"2\"/></font></fonts>"
        "<fills count=\"2\"><fill><patternFill patternType=\"none\"/></fill><fill><patternFill patternType=\"gray125\"/></fill></fills>"
        "<borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders>"
        "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>"
        "<cellXfs count=\"2\">"
        "<xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/>"
        "<xf numFmtId=\"49\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyNumberFormat=\"1\" applyAlignment=\"1\">"
        "<alignment horizontal=\"center\" vertical=\"center\"/>"
        "</xf>"
        "</cellXfs>"
        "<cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles>"
        "</styleSheet>");

    ostringstream sheet;
    sheet << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n";
    sheet << "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
          << "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">";
    sheet << "<sheetViews><sheetView workbookViewId=\"0\"/></sheetViews>";
    sheet << "<sheetFormatPr defaultRowHeight=\"18\"/>";
    sheet << "<cols>";
    for (size_t col = 0; col < max_columns; ++col) {
        const double width = static_cast<double>(min<size_t>(max<size_t>(widths[col] + 2, 12), 60));
        sheet << "<col min=\"" << (col + 1) << "\" max=\"" << (col + 1)
              << "\" width=\"" << width << "\" customWidth=\"1\"/>";
    }
    sheet << "</cols>";
    sheet << "<sheetData>";

    for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
        sheet << "<row r=\"" << (row_idx + 1) << "\">";
        for (size_t col = 0; col < max_columns; ++col) {
            const string value = (col < rows[row_idx].size()) ? rows[row_idx][col] : "";
            sheet << "<c r=\"" << excel_column_name(col) << (row_idx + 1)
                  << "\" s=\"1\" t=\"inlineStr\"><is><t xml:space=\"preserve\">"
                  << escape_xml(value)
                  << "</t></is></c>";
        }
        sheet << "</row>";
    }

    sheet << "</sheetData></worksheet>";
    write_text_file(temp_dir / "xl" / "worksheets" / "sheet1.xml", sheet.str());

    fs::remove(xlsx_file, ec);

    const string temp_dir_str = escape_powershell_single_quoted(temp_dir.string());
    const string xlsx_file_str = escape_powershell_single_quoted(xlsx_file.string());
    const string command =
        "powershell -NoProfile -ExecutionPolicy Bypass -Command \""
        "$ErrorActionPreference='Stop'; "
        "Add-Type -AssemblyName 'System.IO.Compression.FileSystem'; "
        "[System.IO.Compression.ZipFile]::CreateFromDirectory('"
        + temp_dir_str + "', '" + xlsx_file_str + "');\"";

    const int exit_code = system(command.c_str());
    fs::remove_all(temp_dir, ec);

    return exit_code == 0 && fs::exists(xlsx_file);
}

void read_vec(vector<double>& vec,
              string& name_file)
{
    ifstream vec_file(name_file);

    if (!vec_file.is_open()) {
        throw std::runtime_error("Could not open vector file: " + name_file);
    }

    size_t vec_size;

    vec_file >> vec_size;
    vec.resize(vec_size);

    for (size_t i = 0; i < vec_size; ++i) {
        vec_file >> vec[i];
    }

    vec_file.close();
}

void copy_from_nvidia(sycl::queue& q, sycl::buffer<double>& a_buf, double& a) {
    q.submit([&](sycl::handler& h) {
        sycl::accessor acc(a_buf, h, sycl::read_only);
        h.copy(acc, &a);
    }).wait_and_throw();
}

void copy_from_nvidia(sycl::queue& q, sycl::buffer<double>& a_buf, vector<double>& a) {
    q.submit([&](sycl::handler& h) {
        sycl::accessor acc(a_buf, h, sycl::read_only);
        h.copy(acc, a.data());
    }).wait_and_throw();
}

template <typename SolverFn>
static BenchmarkResult run_benchmark(const string& preconditioner,
                                     const string& device,
                                     vector<double>& x,
                                     vector<double>& b,
                                     int& number_iter,
                                     SolverFn&& solver_fn)
{
    struct timespec ts1, ts2;
    double sec = 0.0;
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
            std::cerr << preconditioner << " failed: " << e.what() << std::endl;
            throw;
        }
        timespec_get(&ts2, TIME_UTC);
        sec = (double(ts2.tv_sec) + double(ts2.tv_nsec) / 1000000000.0)
            - (double(ts1.tv_sec) + double(ts1.tv_nsec) / 1000000000.0);
        full_time += sec;
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

BenchmarkResult test_CG_SYCL_jacobi(CSR_matrix<double>& mat,
                                    vector<double>& x,
                                    vector<double>& b,
                                    string& device,
                                    int& number_iter)
{
    return run_benchmark("Jacobi", device, x, b, number_iter,
        [&](sycl::queue& q, vector<double>& x_test) {
            return CG_SYCL_jacobi(mat, x_test, b, q);
        });
}

BenchmarkResult test_CG_SYCL_IC0(CSR_matrix<double>& mat,
                                 vector<double>& x,
                                 vector<double>& b,
                                 string& device,
                                 int& number_iter)
{
    IC0Preconditioner ic0;
    ic0_factor(mat.take_rows(), mat.take_row_ptr(), mat.take_col_ind(), mat.take_val(), ic0);

    return run_benchmark("IC0", device, x, b, number_iter,
        [&](sycl::queue& q, vector<double>& x_test) {
            return CG_SYCL_IC0(mat, x_test, b, q, ic0);
        });
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
    result.block_size = block_size;
    return result;
}

BenchmarkResult test_CG_SYCL_SPAI(CSR_matrix<double>& mat,
                                  vector<double>& x,
                                  vector<double>& b,
                                  string& device,
                                  int& number_iter)
{
    return run_benchmark("SPAI", device, x, b, number_iter,
        [&](sycl::queue& q, vector<double>& x_test) {
            return CG_SYCL_SPAI(mat, x_test, b, q);
        });
}

BenchmarkResult test_CG_SYCL_polynomial(CSR_matrix<double>& mat,
                                        vector<double>& x,
                                        vector<double>& b,
                                        string& device,
                                        int& number_iter,
                                        size_t degree)
{
    BenchmarkResult result = run_benchmark("Polynomial", device, x, b, number_iter,
        [&](sycl::queue& q, vector<double>& x_test) {
            return CG_SYCL_polynomial(mat, x_test, b, q, degree);
        });
    result.polynomial_degree = degree;
    return result;
}

void append_benchmark_to_xlsx(const std::string& xlsx_path,
                              const std::string& matrix_name,
                              const BenchmarkResult& result)
{
    const string source_path = build_xlsx_source_path(xlsx_path);
    const bool write_header = !ifstream(source_path).good();
    ofstream csv(source_path, ios::app);

    if (!csv.is_open()) {
        cerr << "Cannot open benchmark source file for writing: " << source_path << endl;
        return;
    }

    if (write_header) {
        csv << "timestamp;matrix;preconditioner;device;average_time_sec;iterations;error_pct;block_size;polynomial_degree\n";
    }

    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    tm local_tm{};
    localtime_s(&local_tm, &now_time);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &local_tm);

    csv << timestamp << ';'
        << matrix_name << ';'
        << result.preconditioner << ';'
        << result.device_name << ';'
        << format_double_as_text(result.average_time_sec) << ';'
        << result.iterations << ';'
        << format_double_as_text(result.error_pct) << ';'
        << result.block_size << ';'
        << result.polynomial_degree << '\n';

    csv.close();

    if (!rebuild_xlsx_from_source(xlsx_path)) {
        cerr << "Failed to rebuild XLSX report: " << xlsx_path << endl;
    }
}

double check_result(vector<double> first,
                    vector<double> second) {
    double average = 0.0;
    
    for (size_t i = 0; i < first.size(); ++i) {
        average += fabs(first[i] - second[i]);
    }

    return average / first.size();
}

double check_result(vector<double> x,
                    vector<double> b,
                    CSR_matrix<double>& mat) {
    double res = 0.0;
    double b_norm = scalar_product(b, b);
    
    vector<size_t> row_ptr = mat.take_row_ptr();
    vector<size_t> col_ind = mat.take_col_ind();
    vector<double> val = mat.take_val();
    vector<double> Ax(x.size(), 0);
    
    for (size_t i = 0; i < row_ptr.size() - 1; i++) {
        for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            Ax[i] += val[j] * x[col_ind[j]];
        }
    }
    
    for (size_t i = 0; i < x.size(); i++) {
        Ax[i] -= b[i];
    }
    
    res = scalar_product(Ax, Ax);
    
    return sqrt(res) / sqrt(b_norm) * 100;
}
