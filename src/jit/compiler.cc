// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "jit/compiler.hpp"

#include "cuda_utils.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>

namespace tilefusion::jit {

namespace {
// Generate a random string to use as part of the temp file name
std::string generate_random_string(size_t length) {
    static const char alphanum[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, sizeof(alphanum) - 2);

    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        result += alphanum[dist(gen)];
    }
    return result;
}

std::string exec_cmd(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                  pclose);

    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result;
}

std::string get_hash_key(const std::string& kernel_name,
                         const std::string& cuda_source,
                         const std::vector<std::string>& compile_args) {
    std::stringstream ss;
    ss << kernel_name << "___";
    ss << cuda_source << "___";
    for (const auto& arg : compile_args) {
        ss << arg << " ";
    }

    return ss.str();
}

std::string get_nvcc_path() {
#ifdef NVCC_PATH
    return NVCC_PATH;
#else
    return "nvcc";
#endif
}

}  // anonymous namespace

JitCompiler& JitCompiler::instance() {
    static JitCompiler instance;
    return instance;
}

JitCompiler::JitCompiler() {
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to initialize CUDA driver");
    }

    result = cuCtxGetCurrent(&cuda_context_);
    if (result != CUDA_SUCCESS) {
        CUdevice device;
        result = cuDeviceGet(&device, 0);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get CUDA device");
        }

        result = cuCtxCreate(&cuda_context_, 0, device);
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to create CUDA context");
        }
    }
}

JitCompiler::~JitCompiler() {
    for (auto& [key, module] : module_cache_) {
        cuModuleUnload(module);
    }
}

CUfunction JitCompiler::compile_kernel(
    const std::string& kernel_name, const std::string& cuda_source,
    const std::vector<std::string>& compile_args) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key = get_hash_key(kernel_name, cuda_source, compile_args);

    auto kernel_it = kernel_cache_.find(key);
    if (kernel_it != kernel_cache_.end()) {
        return kernel_it->second;
    }

    try {
        std::string ptx = compile_to_ptx(cuda_source, compile_args);

        CUfunction kernel = load_ptx_and_get_kernel(ptx, kernel_name);

        kernel_cache_[key] = kernel;

        return kernel;
    } catch (const std::exception& e) {
        std::cerr << "Error compiling kernel: " << e.what() << std::endl;
        return nullptr;
    }
}

CUfunction JitCompiler::get_or_compile_kernel(
    const std::string& kernel_name, const std::string& cuda_source,
    const std::vector<std::string>& compile_args) {
    return compile_kernel(kernel_name, cuda_source, compile_args);
}

void JitCompiler::clear_cache() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [key, module] : module_cache_) {
        cuModuleUnload(module);
    }

    module_cache_.clear();
    kernel_cache_.clear();
}

std::string JitCompiler::compile_to_ptx(
    const std::string& cuda_source,
    const std::vector<std::string>& compile_args) {
    // Write the CUDA source to a temporary file
    std::string cu_file = write_to_temp_file(cuda_source, ".cu");

    std::stringstream cmd;
    cmd << get_nvcc_path() << " -ptx ";

    cmd << "-arch=sm_70 ";  // FIXME(ying): Auto-detect this

    for (const auto& arg : compile_args) {
        cmd << arg << " ";
    }

    std::string ptx_file = cu_file.substr(0, cu_file.size() - 3) + ".ptx";
    cmd << cu_file << " -o " << ptx_file;

    std::string output = exec_cmd(cmd.str());

    std::ifstream ptx_stream(ptx_file);
    if (!ptx_stream.good()) {
        throw std::runtime_error("Failed to compile CUDA source: " + output);
    }

    std::stringstream ptx_content;
    ptx_content << ptx_stream.rdbuf();

    std::remove(cu_file.c_str());
    std::remove(ptx_file.c_str());

    return ptx_content.str();
}

CUfunction JitCompiler::load_ptx_and_get_kernel(
    const std::string& ptx, const std::string& kernel_name) {
    std::string module_key = kernel_name + "_" + generate_random_string(10);

    CUmodule module;
    CUresult result = cuModuleLoadData(&module, ptx.c_str());
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to load PTX module");
    }

    CUfunction kernel;
    result = cuModuleGetFunction(&kernel, module, kernel_name.c_str());
    if (result != CUDA_SUCCESS) {
        cuModuleUnload(module);
        throw std::runtime_error("Failed to get kernel function: " +
                                 kernel_name);
    }

    module_cache_[module_key] = module;

    return kernel;
}

std::string JitCompiler::write_to_temp_file(const std::string& content,
                                            const std::string& extension) {
    std::string filename =
        "/tmp/tilefusion_" + generate_random_string(10) + extension;

    std::ofstream out(filename);
    if (!out.good()) {
        throw std::runtime_error("Failed to create temporary file: " +
                                 filename);
    }

    out << content;
    out.close();

    return filename;
}

}  // namespace tilefusion::jit
