// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <vector>

namespace tilefusion {

// Returns the name of the device.
std::string get_device_name() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::stringstream ss(prop.name);
    const char delim = ' ';

    std::string s;
    std::vector<std::string> out;

    while (std::getline(ss, s, delim)) {
        out.push_back(s);
    }

    std::stringstream out_ss;
    int i = 0;
    for (; i < static_cast<int>(out.size()) - 1; ++i) out_ss << out[i] << "_";
    out_ss << out[i];
    return out_ss.str();
}
}  // namespace tilefusion
