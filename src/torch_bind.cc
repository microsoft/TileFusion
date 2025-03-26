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

// Helper function to create CppFunction from kernel
template <typename Func>
torch::CppFunction make_op(Func* kernel_fn) {
    return torch::CppFunction(kernel_fn);
}

// Fixed macro to use correct function names
#define REG_KERNEL(name, func)                             \
    std::move(KernelBuilder(#name))                        \
        .sig(name##_sig)                                   \
        .fn(torch::CppFunction(tilefusion::kernels::func)) \
        .build()

// Collection of all kernel definitions
// Helper macro to define kernel signatures more cleanly
#define DEF_KERNEL_SIG(name, signature) \
    static const char* name##_sig = signature;

// Helper macro to add a kernel to the registry
#define ADD_KERNEL(kernels, name, func) \
    kernels.push_back(REG_KERNEL(name, func))

std::vector<KernelInfo> get_kernel_registry() {
    // Define all kernel signatures
    DEF_KERNEL_SIG(
        scatter_nd,
        "scatter_nd(Tensor(a!) data, Tensor updates, Tensor indices) -> ()");
    DEF_KERNEL_SIG(flash_attention_fwd,
                   R"DOC(flash_attention_fwd(
            Tensor(a!) Q,
            Tensor K, Tensor V, Tensor O,
            int m, int n, int k, int p) -> ()
        )DOC");

    std::vector<KernelInfo> kernels;
    ADD_KERNEL(kernels, scatter_nd, scatter_op);
    ADD_KERNEL(kernels, flash_attention_fwd, flash_attention_op);
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
