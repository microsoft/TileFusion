// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ATen/ATen.h>
#include <torch/script.h>

#include <functional>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace tilefusion {

// This macro is used to export the kernel functions to the Python bindings.
// NOTE: It MUST be used for all kernel functions that are used in the Python
// bindings, otherwise you will get undefined symbol errors when you try to
// import the TileFusion module in Python.
#define TILEFUSION_EXPORT extern "C" __attribute__((visibility("default")))

#define REGISTER_OP(name, schema, func)                                  \
    namespace {                                                          \
    static bool name##_registered = []() {                               \
        tilefusion::KernelRegistry::instance().add_kernel(#name, schema, \
                                                          func);         \
        return true;                                                     \
    }();                                                                 \
    }

template <typename KernelFunc>
struct KernelTraits {
    static void register_impl(torch::Library& m, const char* name,
                              KernelFunc func) {
        m.impl(name, torch::DispatchKey::CUDA, func);
    }
};

struct KernelInfo {
    const char* name;
    const char* schema;
    void* func;
    std::type_index type;
};

class KernelRegistry {
  public:
    static KernelRegistry& instance() {
        static KernelRegistry registry;
        return registry;
    }

    template <typename KernelFunc>
    void add_kernel(const char* name, const char* schema, KernelFunc func) {
        kernels_.push_back(
            {name, schema, reinterpret_cast<void*>(func), typeid(KernelFunc)});
        register_kernel_type<KernelFunc>();
    }

    void register_with_torch(torch::Library& m) const {
        for (const auto& kernel : kernels_) {
            m.def(kernel.schema);
        }
    }

    void register_implementations(torch::Library& m) const {
        for (const auto& kernel : kernels_) {
            auto it = registration_functions_.find(kernel.type);
            if (it != registration_functions_.end()) {
                it->second(m, kernel.name, kernel.func);
            }
        }
    }

  private:
    template <typename KernelFunc>
    void register_kernel_type() {
        registration_functions_[typeid(KernelFunc)] =
            [](torch::Library& m, const char* name, void* func) {
                KernelTraits<KernelFunc>::register_impl(
                    m, name, reinterpret_cast<KernelFunc>(func));
            };
    }

    std::vector<KernelInfo> kernels_;
    std::unordered_map<std::type_index,
                       std::function<void(torch::Library&, const char*, void*)>>
        registration_functions_;
};

}  // namespace tilefusion
