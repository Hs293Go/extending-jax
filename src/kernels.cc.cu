// This file contains the GPU implementation of our op. It's a pretty typical
// CUDA kernel and I make no promises about the quality of the code or the
// choices made therein, but it should get the point accross.

#include <stdexcept>

#include "Eigen/Dense"
#include "extending_jax/kernel_helpers.h"
#include "extending_jax/kernels.h"
#include "extending_jax/quadrotor.h"

namespace model {

namespace {
constexpr auto kBlockDim = 16L;
constexpr auto kMaxCUDAThreads = 1024L;

inline void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename T, Eigen::Index N>
using VectorStack = Eigen::Matrix<T, N, Eigen::Dynamic>;

template <typename T>
__global__ void QuadrotorDynamicsKernel(T const *p_x, T const *p_u, T *p_dx,
                                        std::int32_t max_num_models) {
  using Model = Quadrotor<T>;
  using StateStackCMap = Eigen::Map<const VectorStack<T, Model::kStateSize>>;
  using StateStackMap = Eigen::Map<VectorStack<T, Model::kStateSize>>;
  using InputStackCMap = Eigen::Map<const VectorStack<T, Model::kInputSize>>;

  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_num_models) {
    return;
  }

  const StateStackCMap xs{p_x, Model::kStateSize, max_num_models};
  const InputStackCMap us{p_u, Model::kInputSize, max_num_models};
  StateStackMap dxs{p_dx, Model::kStateSize, max_num_models};

  dxs.col(idx) = Model::ModelDerivatives(xs.col(idx), us.col(idx));
}

template <typename T>
__global__ void QuadrotorPushforwardKernel(T const *p_x, T const *p_u,
                                           T const *p_tx, T const *p_tu,
                                           T *p_jvp,
                                           std::int32_t max_num_models) {
  using Model = Quadrotor<T>;
  using StateStackCMap = Eigen::Map<const VectorStack<T, Model::kStateSize>>;
  using StateStackMap = Eigen::Map<VectorStack<T, Model::kStateSize>>;
  using InputStackCMap = Eigen::Map<const VectorStack<T, Model::kInputSize>>;

  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_num_models) {
    return;
  }

  const StateStackCMap xs{p_x, Model::kStateSize, max_num_models};
  const InputStackCMap us{p_u, Model::kInputSize, max_num_models};
  const StateStackCMap txs{p_tx, Model::kStateSize, max_num_models};
  const InputStackCMap tus{p_tu, Model::kInputSize, max_num_models};
  StateStackMap jvps{p_jvp, Model::kStateSize, max_num_models};

  jvps.col(idx) = Model::ModelPushfoward(xs.col(idx), us.col(idx), txs.col(idx),
                                         tus.col(idx));
}

template <typename T>
inline void EvaluateDynamics(cudaStream_t stream, void **buffers,
                             const char *opaque, std::size_t opaque_len) {
  const auto *d = UnpackDescriptor<Descriptor>(opaque, opaque_len);
  const auto max_num_models = d->size;
  if (max_num_models > kBlockDim * kMaxCUDAThreads) {
    throw std::runtime_error(
        "Number of parallel computatations exceeded the numbers of threads");
  }

  const auto *p_x = static_cast<const T *>(buffers[0]);
  const auto *p_u = static_cast<const T *>(buffers[1]);
  auto *p_dx = static_cast<T *>(buffers[2]);

  const auto grid_dim =
      std::min(kMaxCUDAThreads, (max_num_models + kBlockDim - 1) / kBlockDim);
  QuadrotorDynamicsKernel<T>
      <<<grid_dim, kBlockDim, 0, stream>>>(p_x, p_u, p_dx, max_num_models);

  ThrowIfError(cudaGetLastError());
}

template <typename T>
inline void EvaluatePushforward(cudaStream_t stream, void **buffers,
                                const char *opaque, std::size_t opaque_len) {
  const auto *d = UnpackDescriptor<Descriptor>(opaque, opaque_len);
  const auto max_num_models = d->size;
  if (max_num_models > kBlockDim * kMaxCUDAThreads) {
    throw std::runtime_error(
        "Number of parallel computatations exceeded the numbers of threads");
  }

  const auto *p_x = static_cast<const T *>(buffers[0]);
  const auto *p_u = static_cast<const T *>(buffers[1]);
  const auto *p_tx = static_cast<const T *>(buffers[2]);
  const auto *p_tu = static_cast<const T *>(buffers[3]);
  auto *p_jvp = static_cast<T *>(buffers[4]);
  const auto grid_dim =
      std::min(kMaxCUDAThreads, (max_num_models + kBlockDim - 1) / kBlockDim);

  QuadrotorPushforwardKernel<<<grid_dim, kBlockDim, 0, stream>>>(
      p_x, p_u, p_tx, p_tu, p_jvp, max_num_models);

  ThrowIfError(cudaGetLastError());
}

}  // namespace

void gpu_quadrotor_pushforward_f32(cudaStream_t stream, void **buffers,
                                   const char *opaque, std::size_t opaque_len) {
  EvaluatePushforward<float>(stream, buffers, opaque, opaque_len);
}

void gpu_quadrotor_pushforward_f64(cudaStream_t stream, void **buffers,
                                   const char *opaque, std::size_t opaque_len) {
  EvaluatePushforward<double>(stream, buffers, opaque, opaque_len);
}

void gpu_quadrotor_dynamics_f32(cudaStream_t stream, void **buffers,
                                const char *opaque, std::size_t opaque_len) {
  EvaluateDynamics<float>(stream, buffers, opaque, opaque_len);
}

void gpu_quadrotor_dynamics_f64(cudaStream_t stream, void **buffers,
                                const char *opaque, std::size_t opaque_len) {
  EvaluateDynamics<double>(stream, buffers, opaque, opaque_len);
}

}  // namespace model
