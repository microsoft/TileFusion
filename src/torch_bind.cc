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
    // Only initialize name_ and sig_, fn_ will be set later via fn()
    KernelBuilder(const char* name)
        : name_(name),
          sig_(nullptr)
    // Don't initialize fn_ in constructor
    {}

    KernelBuilder&& sig(const char* s) && {
        sig_ = s;
        return std::move(*this);
    }

    KernelBuilder&& fn(torch::CppFunction f) && {
        fn_ = std::move(f);
        return std::move(*this);
    }

    // Make sure fn_ is set before building
    KernelInfo build() && {
        assert(fn_.has_value());  // Ensure fn was set
        return KernelInfo{
            name_, sig_,
            std::move(*fn_)  // Dereference the optional before moving
        };
    }

  private:
    const char* name_;
    const char* sig_;
    std::optional<torch::CppFunction> fn_;
};

// Simplified kernel registry information
struct KernelRegInfo {
    const char* name;
    const char* sig;
};

// Helper macro to define kernel registration
#define REGISTER_KERNEL(str_name, fn_name)                             \
    else if (strcmp(info.name, str_name) == 0) {                       \
        kernels.push_back(                                             \
            std::move(KernelBuilder(info.name))                        \
                .sig(info.sig)                                         \
                .fn(torch::CppFunction(&tilefusion::kernels::fn_name)) \
                .build());                                             \
    }

// Define all kernels
static const KernelRegInfo kernel_registry_info[] = {
    {"scatter_nd",
     "scatter_nd(Tensor(a!) data, Tensor updates, Tensor indices) -> ()"},
    {"flash_attention",
     R"DOC(flash_attention(
            Tensor(a!) Q,
            Tensor K, Tensor V, Tensor O,
            int m, int n, int k, int p) -> ()
        )DOC"}};

std::vector<KernelInfo> get_kernel_registry() {
    std::vector<KernelInfo> kernels;
    for (const auto& info : kernel_registry_info) {
        if (false) {
        }  // start the else-if chain
        REGISTER_KERNEL("scatter_nd", scatter_nd)
        REGISTER_KERNEL("flash_attention", flash_attention)
        else {
            throw std::runtime_error("Unknown kernel: " +
                                     std::string(info.name));
        }
    }
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
