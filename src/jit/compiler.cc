// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "jit/compiler.hpp"

#include "cuda_info.hpp"
#include "cuda_utils.hpp"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

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
  LOG(INFO) << "Executing command: " << cmd;
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                pclose);

  if (!pipe) {
    LOG(ERROR) << "popen() failed for command: " << cmd;
    throw std::runtime_error("popen() failed!");
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }

  int status = pclose(pipe.release());
  if (status != 0) {
    LOG(WARNING) << "Command failed with status " << status << ": " << cmd;
    LOG(WARNING) << "Output: " << result;
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
}  // namespace

JitCompiler& JitCompiler::instance() {
  static JitCompiler instance;
  return instance;
}

JitCompiler::JitCompiler() {
  // FIXME(ying): GLog should be initialized before this function is called

  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    LOG(FATAL) << "Failed to initialize CUDA driver";
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
    const std::vector<std::string>& include_paths,
    const std::vector<std::string>& compile_args) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string key = get_hash_key(kernel_name, cuda_source, compile_args);

  auto kernel_it = kernel_cache_.find(key);
  if (kernel_it != kernel_cache_.end()) {
    return kernel_it->second;
  }

  try {
    std::string ptx = compile_to_ptx(cuda_source, include_paths, compile_args);
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
    const std::vector<std::string>& include_paths,
    const std::vector<std::string>& compile_args) {
  return compile_kernel(kernel_name, cuda_source, include_paths, compile_args);
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
    const std::vector<std::string>& include_paths,
    const std::vector<std::string>& compile_args) {
  std::string cu_file = write_to_temp_file(cuda_source, ".cu");

  std::stringstream cmd;
  cmd << get_nvcc_path() << " -ptx ";

  cmd << "-arch=" << GetComputeCapability() << " ";

  for (const auto& path : include_paths) {
    cmd << "-I" << path << " ";
  }

  for (const auto& arg : compile_args) {
    cmd << arg << " ";
  }

  std::string ptx_file = cu_file.substr(0, cu_file.size() - 3) + ".ptx";
  cmd << cu_file << " -o " << ptx_file;

  std::string output = exec_cmd(cmd.str());

  std::ifstream ptx_stream(ptx_file);
  if (!ptx_stream.good()) {
    LOG(ERROR) << "Failed to open PTX file: " << ptx_file;
    LOG(ERROR) << "nvcc output: " << output;
    throw std::runtime_error("Failed to compile CUDA source: " + output);
  }

  std::stringstream ptx_content;
  ptx_content << ptx_stream.rdbuf();
  ptx_stream.close();

  // For debugging, keep the files instead of removing them
  LOG(INFO) << "Keeping temporary files for debugging:";
  LOG(INFO) << "  Source: " << cu_file;
  LOG(INFO) << "  PTX: " << ptx_file;

  // Comment out the removal for debugging
  // std::remove(cu_file.c_str());
  // std::remove(ptx_file.c_str());

  return ptx_content.str();
}

CUfunction JitCompiler::load_ptx_and_get_kernel(
    const std::string& ptx, const std::string& kernel_name) {
  std::string module_key = kernel_name + "_" + generate_random_string(10);

  CUmodule module;
  CUDA_DRIVER_CHECK(cuModuleLoadData(&module, ptx.c_str()));

  CUfunction kernel;
  CUDA_DRIVER_CHECK(cuModuleGetFunction(&kernel, module, kernel_name.c_str()));

  module_cache_[module_key] = module;
  LOG(INFO) << "Loaded PTX module: " << module_key;
  return kernel;
}

std::string JitCompiler::write_to_temp_file(const std::string& content,
                                            const std::string& extension) {
  const char* home_dir = getenv("HOME");
  std::string cache_dir;

  if (!home_dir || strlen(home_dir) == 0) {
    LOG(WARNING)
        << "HOME environment variable not set or empty, using /tmp instead";
    cache_dir = "/tmp/tilefusion";
  } else {
    cache_dir = std::string(home_dir) + "/.cache/tilefusion";
  }

  std::string mkdir_cmd = "mkdir -p " + cache_dir;

  int ret = system(mkdir_cmd.c_str());
  if (ret != 0) {
    LOG(ERROR) << "Failed to create cache directory (ret=" << ret
               << "): " << cache_dir;
    throw std::runtime_error("Failed to create cache directory: " + cache_dir);
  }

  std::string filename =
      cache_dir + "/" + generate_random_string(10) + extension;

  std::ofstream out(filename);
  if (!out.good()) {
    LOG(ERROR) << "Failed to open file for writing: " << filename;
    throw std::runtime_error("Failed to create temporary file: " + filename);
  }

  out << content;
  if (!out.good()) {
    LOG(ERROR) << "Failed to write content to file: " << filename;
    throw std::runtime_error("Failed to write to temporary file: " + filename);
  }

  out.close();
  if (!out) {
    LOG(ERROR) << "Failed to close file: " << filename;
    throw std::runtime_error("Failed to close temporary file: " + filename);
  }

  std::ifstream check(filename);
  if (!check.good()) {
    LOG(ERROR) << "File verification failed: " << filename;
    throw std::runtime_error("File doesn't exist after write: " + filename);
  }
  check.close();

  return filename;
}

}  // namespace tilefusion::jit
