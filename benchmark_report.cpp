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
        current /= 26;
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

    fs::path temp_xlsx_file = xlsx_file;
    temp_xlsx_file += ".tmp";
    fs::remove(temp_xlsx_file, ec);

    const string temp_dir_str = escape_powershell_single_quoted(temp_dir.string());
    const string xlsx_file_str = escape_powershell_single_quoted(temp_xlsx_file.string());
    const string command =
        "powershell -NoProfile -ExecutionPolicy Bypass -Command \""
        "$ErrorActionPreference='Stop'; "
        "Add-Type -AssemblyName 'System.IO.Compression.FileSystem'; "
        "[System.IO.Compression.ZipFile]::CreateFromDirectory('"
        + temp_dir_str + "', '" + xlsx_file_str + "');\"";

    const int exit_code = system(command.c_str());
    fs::remove_all(temp_dir, ec);

    if (exit_code != 0 || !fs::exists(temp_xlsx_file)) {
        return false;
    }

    fs::remove(xlsx_file, ec);
    fs::rename(temp_xlsx_file, xlsx_file, ec);
    return !ec && fs::exists(xlsx_file);
}

static void migrate_benchmark_source_if_needed(const string& source_path)
{
    vector<vector<string>> rows = read_semicolon_table(source_path);
    if (rows.empty()) {
        return;
    }

    const vector<string> target_header = {
        "timestamp",
        "matrix",
        "preconditioner",
        "device",
        "matrix_rows",
        "matrix_cols",
        "matrix_nnz",
        "average_time_sec",
        "iterations",
        "error_pct",
        "block_size",
        "polynomial_degree",
        "library"
    };

    if (rows[0] == target_header) {
        return;
    }

    const vector<string>& old_header = rows[0];
    // Detect prior schema (without matrix_nnz): 12 cols ending in "library", with
    // "matrix_rows"/"matrix_cols" present but no "matrix_nnz".
    const bool schema_12 =
        old_header.size() == 12 &&
        std::find(old_header.begin(), old_header.end(), std::string("matrix_nnz")) == old_header.end() &&
        old_header.back() == "library";
    const bool schema_pre_rows =
        old_header.size() >= 9 && old_header.size() < 12 &&
        std::find(old_header.begin(), old_header.end(), std::string("matrix_rows")) == old_header.end();
    const bool has_library_in_old = old_header.size() >= 10 && old_header.back() == "library";
    rows[0] = target_header;
    for (size_t i = 1; i < rows.size(); ++i) {
        vector<string> migrated;
        migrated.reserve(target_header.size());

        if (schema_12 && rows[i].size() >= 12) {
            // timestamp;matrix;preconditioner;device;matrix_rows;matrix_cols;
            //   average_time_sec;iterations;error_pct;block_size;polynomial_degree;library
            migrated.push_back(rows[i][0]);
            migrated.push_back(rows[i][1]);
            migrated.push_back(rows[i][2]);
            migrated.push_back(rows[i][3]);
            migrated.push_back(rows[i][4]);
            migrated.push_back(rows[i][5]);
            migrated.push_back("0"); // matrix_nnz (unknown for old rows)
            migrated.push_back(rows[i][6]);
            migrated.push_back(rows[i][7]);
            migrated.push_back(rows[i][8]);
            migrated.push_back(rows[i][9]);
            migrated.push_back(rows[i][10]);
            migrated.push_back(rows[i][11]);
        } else if (schema_pre_rows && rows[i].size() >= 9) {
            // Very old schema without matrix_rows/matrix_cols
            migrated.push_back(rows[i][0]);
            migrated.push_back(rows[i][1]);
            migrated.push_back(rows[i][2]);
            migrated.push_back(rows[i][3]);
            migrated.push_back("0");
            migrated.push_back("0");
            migrated.push_back("0"); // matrix_nnz
            migrated.push_back(rows[i][4]);
            migrated.push_back(rows[i][5]);
            migrated.push_back(rows[i][6]);
            migrated.push_back(rows[i][7]);
            migrated.push_back(rows[i][8]);
            migrated.push_back(has_library_in_old && rows[i].size() >= 10 ? rows[i][9] : "Custom");
        } else {
            migrated = rows[i];
            while (migrated.size() < target_header.size()) {
                migrated.push_back("0");
            }
            if (migrated.size() > target_header.size()) {
                migrated.resize(target_header.size());
            }
            migrated.back() = (has_library_in_old && rows[i].size() >= target_header.size())
                ? rows[i].back()
                : "Custom";
        }

        rows[i] = migrated;
    }

    ofstream output(source_path, ios::trunc);
    for (const auto& row : rows) {
        for (size_t col = 0; col < row.size(); ++col) {
            if (col != 0) {
                output << ';';
            }
            output << row[col];
        }
        output << '\n';
    }
}

void append_benchmark_to_xlsx(const std::string& xlsx_path,
                              const std::string& matrix_name,
                              const BenchmarkResult& result)
{
    const string source_path = build_xlsx_source_path(xlsx_path);
    if (ifstream(source_path).good()) {
        migrate_benchmark_source_if_needed(source_path);
    }
    const bool write_header = !ifstream(source_path).good();
    ofstream csv(source_path, ios::app);

    if (!csv.is_open()) {
        cerr << "Cannot open benchmark source file for writing: " << source_path << endl;
        return;
    }

    if (write_header) {
        csv << "timestamp;matrix;preconditioner;device;matrix_rows;matrix_cols;matrix_nnz;average_time_sec;iterations;error_pct;block_size;polynomial_degree;library\n";
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
        << result.matrix_rows << ';'
        << result.matrix_cols << ';'
        << result.matrix_nnz << ';'
        << format_double_as_text(result.average_time_sec) << ';'
        << result.iterations << ';'
        << format_double_as_text(result.error_pct) << ';'
        << result.block_size << ';'
        << result.polynomial_degree << ';'
        << result.library << '\n';

    csv.close();

    if (!rebuild_xlsx_from_source(xlsx_path)) {
        cerr << "Failed to rebuild XLSX report: " << xlsx_path << endl;
    }
}
