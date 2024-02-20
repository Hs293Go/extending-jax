#ifndef EXTENDING_JAX_KERNELS_H_
#define EXTENDING_JAX_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace model {
struct Descriptor {
  std::int64_t size;
};

void gpu_quadrotor_dynamics_f32(cudaStream_t stream, void** buffers,
                                const char* opaque, std::size_t opaque_len);
void gpu_quadrotor_dynamics_f64(cudaStream_t stream, void** buffers,
                                const char* opaque, std::size_t opaque_len);
void gpu_quadrotor_pushforward_f32(cudaStream_t stream, void** buffers,
                                   const char* opaque, std::size_t opaque_len);
void gpu_quadrotor_pushforward_f64(cudaStream_t stream, void** buffers,
                                   const char* opaque, std::size_t opaque_len);

}  // namespace model

#endif  // EXTENDING_JAX_KERNELS_H_
