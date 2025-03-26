// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kernels/mod.hpp"

namespace tilefusion {
using namespace tilefusion::kernels;

namespace {
// Helper struct to hold kernel information
struct KernelInfo {
    const char* name;
    const char* sig;        // shorter than 'signature'
    torch::CppFunction fn;  // shorter than 'impl'
};

// Builder class for kernel registration
class KernelBuilder {
  public:
    // Initialize without fn_ in constructor
    KernelBuilder(const char* name) : name_(name) {}

    KernelBuilder& sig(const char* s) {
        sig_ = s;
        return *this;
    }

    // Take CppFunction by value and move it
    KernelBuilder& fn(torch::CppFunction&& f) {
        fn_ = std::move(f);
        return *this;
    }

    KernelInfo build() && { return {name_, sig_, std::move(fn_)}; }

  private:
    const char* name_;
    const char* sig_;
    torch::CppFunction fn_{torch::CppFunction(
        nullptr, nullptr, nullptr)};  // Initialize with dummy values
};

// Fixed macro to use _op suffix and proper function wrapping
#define REG_KERNEL(name, signature)                          \
    KernelBuilder(#name)                                     \
        .sig(signature)                                      \
        .fn(torch::CppFunction(name##_op, nullptr, nullptr)) \
        .build()

// Collection of all kernel definitions
std::vector<KernelInfo> get_kernel_registry() {
    static const char* scatter_sig =
        "scatter_nd(Tensor(a!) data, Tensor updates, Tensor indices) -> ()";
    static const char* flash_attn_sig =
        R"DOC(flash_attention_fwd(
            Tensor(a!) Q,
            Tensor K, Tensor V, Tensor O,
            int m, int n, int k, int p) -> ()
        )DOC";

    std::vector<KernelInfo> kernels;
    kernels.push_back(REG_KERNEL(scatter_nd, scatter_sig));
    kernels.push_back(REG_KERNEL(flash_attention_fwd, flash_attn_sig));
    return kernels;
}

// Single function to handle both registrations
void register_kernels(torch::Library& m, bool is_impl) {
    for (auto&& k : get_kernel_registry()) {
        if (is_impl) {
            m.impl(k.name, std::move(k.fn));
        } else {
            m.def(k.sig);
        }
    }
}
}  // anonymous namespace

TORCH_LIBRARY_IMPL(tilefusion, CUDA, m) { register_kernels(m, true); }

TORCH_LIBRARY(tilefusion, m) { register_kernels(m, false); }
}  // namespace tilefusion
