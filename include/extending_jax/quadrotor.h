#ifndef EXTENDING_JAX_QUADROTOR_H_
#define EXTENDING_JAX_QUADROTOR_H_

#include <eigen3/Eigen/Core>

#include "Eigen/Dense"

template <typename T>
struct Quadrotor {
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
  EIGEN_DEVICE_FUNC static StateType ModelDerivatives(
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

    const Scalar f = u[0];
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
  EIGEN_DEVICE_FUNC static StateType ModelPushfoward(
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
#endif  // EXTENDING_JAX_QUADROTOR_H_
