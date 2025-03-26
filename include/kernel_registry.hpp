#pragma once

#include <ATen/ATen.h>
#include <torch/script.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// Define export macro
#if defined(_MSC_VER)
    #define TILEFUSION_EXPORT __declspec(dllexport)
#else
    #define TILEFUSION_EXPORT __attribute__((visibility("default")))
#endif

namespace tilefusion {

// Function traits to handle different kernel signatures
template <typename T>
struct KernelTraits;

// Specialization for scatter_nd
template <>
struct KernelTraits<void(at::Tensor&, const at::Tensor&, const at::Tensor&)> {
    static void register_impl(torch::Library& m, const char* name, void* fn) {
        using FuncType =
            void (*)(at::Tensor&, const at::Tensor&, const at::Tensor&);
        m.impl(name, reinterpret_cast<FuncType>(fn));
    }
};

// Simple struct to hold kernel info
struct KernelInfo {
    const char* name;
    const char* schema;
    void* func;
};

// Simple registry class
class TILEFUSION_EXPORT KernelRegistry {
  public:
    static KernelRegistry& instance() {
        static KernelRegistry registry;
        return registry;
    }

    void add_kernel(const char* name, const char* schema, void* func) {
        kernels_.push_back({name, schema, func});
    }

    void register_with_torch(torch::Library& m) const {
        for (const auto& kernel : kernels_) {
            m.def(kernel.schema);
        }
    }

    void register_implementations(torch::Library& m) const {
        for (const auto& kernel : kernels_) {
            using ScatterFunc =
                void (*)(at::Tensor&, const at::Tensor&, const at::Tensor&);
            m.impl(kernel.name, reinterpret_cast<ScatterFunc>(kernel.func));
        }
    }

  private:
    std::vector<KernelInfo> kernels_;
};

}  // namespace tilefusion

// Simple registration macro
#define REGISTER_KERNEL(name, schema, func)                              \
    namespace {                                                          \
    static bool name##_registered = []() {                               \
        tilefusion::KernelRegistry::instance().add_kernel(#name, schema, \
                                                          (void*)func);  \
        return true;                                                     \
    }();                                                                 \
    }
