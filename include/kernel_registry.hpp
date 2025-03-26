#pragma once

#include "kernels/mod.hpp"

#include <torch/custom_class.h>

namespace tilefusion {

// Helper struct to hold kernel information
struct KernelInfo {
    const char* name;
    const char* sig;
    torch::CppFunction fn;
};

struct KernelRegEntry {
    const char* name;
    const char* sig;
    void* fn_ptr;  // Store raw function pointer instead of CppFunction
};

// Helper macro to define a kernel entry
#define DEFINE_KERNEL(name, signature)            \
    static const KernelRegEntry name##_kernel = { \
        #name, signature, (void*)&tilefusion::kernels::name}

}  // namespace tilefusion
