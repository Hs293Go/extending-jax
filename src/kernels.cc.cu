// This file contains the GPU implementation of our op. It's a pretty typical
// CUDA kernel and I make no promises about the quality of the code or the
// choices made therein, but it should get the point accross.

#include "Eigen/Dense"
#include "extending_jax/kernel_helpers.h"
#include "extending_jax/kernels.h"

namespace model {

namespace {
constexpr auto kBlockDim = 8L;
constexpr auto kMaxCUDAThreads = 1024L;

inline void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename T>
struct QuadrotorDynamics {
  enum { kStateSize = 10, kInputSize = 4 };

  using StateType = Eigen::Matrix<T, kStateSize, 1>;
  using StateJacobian = Eigen::Matrix<T, kStateSize, kStateSize>;
  using InputType = Eigen::Matrix<T, kInputSize, 1>;

  struct PushFowardResult {
    StateType by_state;
    StateType by_input;
  };

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

    constexpr auto kGravAccel = Scalar{-9.81};

    Eigen::Map<const QuaternionType> q(x.template segment<4>(3).data());
    Eigen::Ref<const VectorType> v(x.template tail<3>());

    const T f = u[0];
    const QuaternionType omega_q{Scalar{0}, u[1] / Scalar{2}, u[2] / Scalar{2},
                                 u[3] / Scalar{2}};

    StateType dx;
    dx.template head<3>() = v;
    dx.template segment<4>(3) = (q * omega_q).coeffs();
    dx.template tail<3>() =
        q * VectorType::UnitZ() * f + VectorType::UnitZ() * kGravAccel;
    return dx;
  }

  // WIP(Hs293Go): Bind this into python as well
  template <typename SDerived, typename UDerived, typename TSDerived,
            typename TUDerived>
  __host__ __device__ static StateType ModelPushfoward(
      const Eigen::MatrixBase<SDerived> &x,
      const Eigen::MatrixBase<UDerived> &u,
      const Eigen::MatrixBase<TSDerived> &tx,
      const Eigen::MatrixBase<TUDerived> &tu) {
    Eigen::Map<const QuaternionType> q(x.template segment<4>(3).data());
    Eigen::Ref<const VectorType> v(x.template tail<3>());

    const Scalar f = u[0];
    const QuaternionType omega_q{Scalar{0}, u[1] / Scalar{2}, u[2] / Scalar{2},
                                 u[3] / Scalar{2}};

    // tangent_x[3:5]
    Eigen::Ref<const VectorType> tx_36{tx.template segment<3>(3)};

    // tangent_x[3:6] treated as a quaternion
    Eigen::Map<const QuaternionType> tx_36_q{tx.template segment<4>(3).data()};

    // vector part of the attitude quaternion crossed with unit Z
    const VectorType qv_x_i3{q.y(), -q.x(), Scalar{0}};

    // unit Z crossed with tangent_x[3:5]
    const VectorType i3_x_tx_35{-tx_36.y(), tx_36.x(), Scalar{0}};

    const Scalar qv_dot_i3 = q.z();
    QuaternionType ux_14_q{Scalar{0}, tu[1] / Scalar{2}, tu[2] / Scalar{2},
                           tu[3] / Scalar{2}};
    StateType res1;
    res1.template head<3>() = tx.template tail<3>();
    res1.template segment<4>(3) =
        (tx_36_q * omega_q).coeffs() + (q * ux_14_q).coeffs();
    res1.template tail<3>() =
        Scalar{2} * f *
            (qv_dot_i3 * tx_36 - qv_x_i3.cross(tx_36) - q.w() * i3_x_tx_35 +
             (qv_x_i3 + q.w() * VectorType::UnitZ()) * tx[6]) +
        q * VectorType::UnitZ() * tu[0];

    return res1;
  }
};

template <typename T, Eigen::Index N>
using VectorStack = Eigen::Matrix<T, N, Eigen::Dynamic>;

template <typename T>
__global__ void QuadrotorDynamicsKernel(T const *p_x, T const *p_u, T *p_dx,
                                        std::int32_t max_num_models) {
  using Model = QuadrotorDynamics<T>;
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
  using Model = QuadrotorDynamics<T>;
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
inline void EvaluatePushForward(cudaStream_t stream, void **buffers,
                                const char *opaque, std::size_t opaque_len) {
  const auto *d = UnpackDescriptor<Descriptor>(opaque, opaque_len);
  const auto max_num_models = d->size;

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
  EvaluatePushForward<float>(stream, buffers, opaque, opaque_len);
}

void gpu_quadrotor_pushforward_f64(cudaStream_t stream, void **buffers,
                                   const char *opaque, std::size_t opaque_len) {
  EvaluatePushForward<double>(stream, buffers, opaque, opaque_len);
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
