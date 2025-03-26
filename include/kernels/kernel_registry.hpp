#pragma once
#include <ATen/ATen.h>

#include <string>
#include <vector>

namespace tilefusion {

struct KernelInfo {
    const char* name;
    void* func_ptr;
    const char* schema;
};

class KernelRegistry {
  public:
    static KernelRegistry& instance() {
        static KernelRegistry registry;
        return registry;
    }

    void register_kernel(const char* name, void* func_ptr, const char* schema) {
        kernels_.push_back({name, func_ptr, schema});
    }

    void* get_kernel(const char* name) {
        for (const auto& kernel : kernels_) {
            if (strcmp(kernel.name, name) == 0) {
                return kernel.func_ptr;
            }
        }
        return nullptr;
    }

  private:
    KernelRegistry() = default;
    std::vector<KernelInfo> kernels_;
};

#define REGISTER_KERNEL(name, func, schema)                     \
    static bool _registered_##name = []() {                     \
        tilefusion::KernelRegistry::instance().register_kernel( \
            #name, (void*)func, schema);                        \
        return true;                                            \
    }();

}  // namespace tilefusion
