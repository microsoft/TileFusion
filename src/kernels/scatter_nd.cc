#include "kernels/scatter_nd.hpp"

#include <ATen/ATen.h>

namespace tilefusion {
namespace kernels {

extern "C" {
TILEFUSION_EXPORT
void scatter_nd(at::Tensor& data, const at::Tensor& updates,
                const at::Tensor& indices) {}
}

}  // namespace kernels
}  // namespace tilefusion
