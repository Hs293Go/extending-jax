// This file contains the GPU implementation of our op. It's a pretty typical
// CUDA kernel and I make no promises about the quality of the code or the
// choices made therein, but it should get the point accross.

#include "Eigen/Dense"
#include "extending_jax/kernel_helpers.h"
#include "extending_jax/kernels.h"

namespace kepler_jax {

namespace {

template <typename T>
struct QuadrotorDynamics {
  enum { kStateSize = 10, kInputSize = 4 };

  using StateType = Eigen::Matrix<T, kStateSize, 1>;
  using StateJacobian = Eigen::Matrix<T, kStateSize, kStateSize>;
  using InputType = Eigen::Matrix<T, kInputSize, 1>;

  using Scalar = T;
  using VectorType = Eigen::Matrix<T, 3, 1>;
  using QuaternionType = Eigen::Quaternion<T>;

  template <typename SDerived, typename UDerived>
  __host__ __device__ static StateType ModelDerivatives(
      const Eigen::MatrixBase<SDerived> &x,
      const Eigen::MatrixBase<UDerived> &u) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(SDerived, kStateSize);
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(UDerived, kInputSize);
    static_assert(std::is_same_v<typename SDerived::Scalar, Scalar> &&
                      std::is_same_v<typename UDerived::Scalar, Scalar>,
                  "Scalar type mismatch");

    constexpr auto kGravAccel = T(-9.81);

    Eigen::Map<const QuaternionType> orientation(
        x.template segment<4>(3).data());
    Eigen::Ref<const VectorType> velocity(x.template tail<3>());

    const T thrust = u[0];
    const QuaternionType body_rates_q{T(0), u[1] / T(2), u[2] / T(2),
                                      u[3] / T(2)};

    StateType dx;
    dx.template head<3>() = velocity;
    dx.template segment<4>(3) = (orientation * body_rates_q).coeffs();
    dx.template tail<3>() = orientation * VectorType::UnitZ() * thrust +
                            VectorType::UnitZ() * kGravAccel;
    return dx;
  }

  template <typename SDerived, typename UDerived>
  __host__ __device__ static StateJacobian ModelDerivativeJacobian(
      const Eigen::MatrixBase<SDerived> &x,
      const Eigen::MatrixBase<UDerived> &u) {
    StateJacobian jac = StateJacobian::Zero();
    jac.template block<3, 3>(0, 7).setIdentity();
  }
};

template <typename T, Eigen::Index N>
using VectorStack = Eigen::Matrix<T, N, Eigen::Dynamic>;

template <typename T>
__global__ static void QuadrotorDynamicsKernel(T const *p_x_0, T const *p_u_0,
                                               T *p_dx_0,
                                               std::int32_t max_num_models) {
  using Model = QuadrotorDynamics<T>;
  using StateStackCMap = Eigen::Map<const VectorStack<T, Model::kStateSize>>;
  using StateStackMap = Eigen::Map<VectorStack<T, Model::kStateSize>>;
  using InputStackCMap = Eigen::Map<const VectorStack<T, Model::kInputSize>>;

  const auto idx_model = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_model >= max_num_models) {
    return;
  }

  const StateStackCMap x_stack(p_x_0, Model::kStateSize, max_num_models);
  const InputStackCMap u_stack(p_u_0, Model::kInputSize, max_num_models);
  StateStackMap dx_stack(p_dx_0, Model::kStateSize, max_num_models);

  dx_stack.col(idx_model) =
      Model::ModelDerivatives(x_stack.col(idx_model), u_stack.col(idx_model));
}

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename T>
inline void InvokeKernel(cudaStream_t stream, void **buffers,
                         const char *opaque, std::size_t opaque_len) {
  const auto *d = UnpackDescriptor<KeplerDescriptor>(opaque, opaque_len);
  const auto size = d->size;

  const auto *p_x = static_cast<const T *>(buffers[0]);
  const auto *p_u = static_cast<const T *>(buffers[1]);
  auto *p_dx = static_cast<T *>(buffers[2]);

  const int block_dim = 16;
  constexpr auto kMaxCUDAThreads = 1024;
  const int grid_dim =
      std::min<int>(kMaxCUDAThreads, (size + block_dim - 1) / block_dim);
  QuadrotorDynamicsKernel<T>
      <<<grid_dim, block_dim, 0, stream>>>(p_x, p_u, p_dx, size);

  ThrowIfError(cudaGetLastError());
}

}  // namespace

void gpu_kepler_f32(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  InvokeKernel<float>(stream, buffers, opaque, opaque_len);
}

void gpu_kepler_f64(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  InvokeKernel<double>(stream, buffers, opaque, opaque_len);
}

}  // namespace kepler_jax
