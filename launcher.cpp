#include <windows.h>

#include <string>
#include <vector>

int wmain(int argc, wchar_t* argv[])
{
    wchar_t module_path_buffer[MAX_PATH];
    const DWORD module_len = GetModuleFileNameW(nullptr, module_path_buffer, MAX_PATH);
    if (module_len == 0 || module_len == MAX_PATH) {
        return 1;
    }

    std::wstring root_dir(module_path_buffer);
    const size_t last_sep = root_dir.find_last_of(L"\\/");
    if (last_sep == std::wstring::npos) {
        return 1;
    }
    root_dir.resize(last_sep);

    const std::wstring runtime_dir = root_dir + L"\\runtime";
    const std::wstring app_path = runtime_dir + L"\\sycl_app_real.exe";

    const DWORD app_attributes = GetFileAttributesW(app_path.c_str());
    if (app_attributes == INVALID_FILE_ATTRIBUTES || (app_attributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        return 1;
    }

    std::wstring command_line = L"\"" + app_path + L"\"";
    for (int i = 1; i < argc; ++i) {
        command_line += L" \"";
        command_line += argv[i];
        command_line += L"\"";
    }

    wchar_t* current_path = nullptr;
    size_t current_path_len = 0;
    _wdupenv_s(&current_path, &current_path_len, L"PATH");
    std::wstring new_path = runtime_dir;
    if (current_path != nullptr && current_path[0] != L'\0') {
        new_path += L";";
        new_path += current_path;
    }
    _wputenv_s(L"PATH", new_path.c_str());
    free(current_path);

    STARTUPINFOW startup_info{};
    startup_info.cb = sizeof(startup_info);
    PROCESS_INFORMATION process_info{};

    std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back(L'\0');

    const BOOL created = CreateProcessW(
        app_path.c_str(),
        mutable_command.data(),
        nullptr,
        nullptr,
        TRUE,
        0,
        nullptr,
        root_dir.c_str(),
        &startup_info,
        &process_info);

    if (!created) {
        return static_cast<int>(GetLastError());
    }

    WaitForSingleObject(process_info.hProcess, INFINITE);

    DWORD exit_code = 1;
    GetExitCodeProcess(process_info.hProcess, &exit_code);

    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);

    return static_cast<int>(exit_code);
}
