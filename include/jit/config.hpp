#pragma once

#include <string>
#include <vector>

namespace tilefusion::jit {
// Default JIT include paths
inline std::vector<std::string> get_default_include_paths() {
    std::string current_file = __FILE__;
    std::string project_root =
        current_file.substr(0, current_file.find("/include/"));
    return {project_root + "/include",
            project_root + "/3rd-party/cutlass/include"};
}

// Default JIT compilation arguments
inline std::vector<std::string> get_default_compile_args() {
    return {"-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DNDEBUG",
            "-Xcompiler",
            "-fPIC",
            "-Xcompiler",
            "-Wall",
            "-Xcompiler",
            "-Wextra"};
}
}  // namespace tilefusion::jit
