// This file defines the Python interface to the XLA custom call implemented on
// the CPU. It is exposed as a standard pybind11 module defining "capsule"
// objects containing our method. For simplicity, we export a separate capsule
// for each supported dtype.

#include <algorithm>
#include <future>
#include <iostream>
#include <thread>

#include "extending_jax/pybind11_kernel_helpers.h"
#include "extending_jax/quadrotor.h"

using namespace model;

template <typename Fn>
void ForILoop(std::int64_t low, std::int64_t high, Fn fcn) {
  for (; low < high; ++low) {
    fcn(low);
  }
}

template <typename Fn>
void ParallelForILoop(std::int64_t low, std::int64_t high, Fn fcn,
                      std::int64_t num_threads = 12l) {
  if (low >= high) {
    return;
  }
  const auto length = high - low;
  const auto block_size = length / num_threads;

  std::vector<std::future<void>> futures;
  futures.reserve(num_threads);
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  struct Joiner {
    Joiner(std::vector<std::thread> &t) : threads(t) {}

    ~Joiner() {
      for (auto &&it : threads) {
        if (it.joinable()) {
          it.join();
        }
      }
    }

    std::vector<std::thread> &threads;
  } join_threads(threads);

  auto block_start = 0l;
  for (auto i = 0l; i < num_threads - 1; ++i) {
    auto block_end = block_start;
    block_end += block_size;

    std::packaged_task<void()> task([fcn, block_start, block_end] {
      ForILoop(block_start, block_end, fcn);
    });
    futures.emplace_back(task.get_future());
    threads.emplace_back(std::move(task));
  }

  ForILoop(block_start, length, fcn);

  for (auto &&it : futures) {
    it.get();
  }
}

namespace {
template <typename T, Eigen::Index N>
using VectorStack = Eigen::Matrix<T, N, Eigen::Dynamic>;

template <typename T>
void EvaluateDynamics(void *out_tuple, const void **in) {
  using Model = Quadrotor<T>;
  using StateStackCMap = Eigen::Map<const VectorStack<T, Model::kStateSize>>;
  using StateStackMap = Eigen::Map<VectorStack<T, Model::kStateSize>>;
  using InputStackCMap = Eigen::Map<const VectorStack<T, Model::kInputSize>>;

  // Parse the inputs
  const std::int64_t max_num_models = *static_cast<const std::int64_t *>(in[0]);
  const T *p_x = static_cast<const T *>(in[1]);
  const T *p_u = static_cast<const T *>(in[2]);

  // The output is stored as a list of pointers since we have multiple outputs
  T *p_dx = static_cast<T *>(out_tuple);
  const StateStackCMap xs{p_x, Model::kStateSize, max_num_models};
  const InputStackCMap us{p_u, Model::kInputSize, max_num_models};
  StateStackMap dxs{p_dx, Model::kStateSize, max_num_models};

  constexpr auto kMinPerThread = 8l;
  const auto max_threads = (max_num_models + kMinPerThread - 1) / kMinPerThread;

  if (max_threads > 1) {
    const auto hardware_threads =
        static_cast<std::int64_t>(std::thread::hardware_concurrency());

    const auto num_threads =
        std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);

    ParallelForILoop(
        0, max_num_models,
        [&xs, &us, &dxs](auto j) {
          dxs.col(j) = Model::ModelDerivatives(xs.col(j), us.col(j));
        },
        num_threads);

  } else {
    for (auto j = 0; j != max_num_models; ++j) {
      dxs.col(j) = Model::ModelDerivatives(xs.col(j), us.col(j));
    }
  }
}

template <typename T>
void EvaluatePushforward(void *out_tuple, const void **in) {
  using Model = Quadrotor<T>;
  using StateStackCMap = Eigen::Map<const VectorStack<T, Model::kStateSize>>;
  using StateStackMap = Eigen::Map<VectorStack<T, Model::kStateSize>>;
  using InputStackCMap = Eigen::Map<const VectorStack<T, Model::kInputSize>>;

  // Parse the inputs
  const std::int64_t max_num_models = *static_cast<const std::int64_t *>(in[0]);
  const T *p_x = static_cast<const T *>(in[1]);
  const T *p_u = static_cast<const T *>(in[2]);
  const T *p_tx = static_cast<const T *>(in[3]);
  const T *p_tu = static_cast<const T *>(in[4]);

  // The output is stored as a list of pointers since we have multiple outputs
  T *p_dx = static_cast<T *>(out_tuple);
  const StateStackCMap xs{p_x, Model::kStateSize, max_num_models};
  const InputStackCMap us{p_u, Model::kInputSize, max_num_models};
  const StateStackCMap txs{p_tx, Model::kStateSize, max_num_models};
  const InputStackCMap tus{p_tu, Model::kInputSize, max_num_models};
  StateStackMap jvps{p_dx, Model::kStateSize, max_num_models};

  constexpr auto kMinPerThread = 8l;
  const auto max_threads = (max_num_models + kMinPerThread - 1) / kMinPerThread;
  if (max_threads > 1) {
    const auto hardware_threads =
        static_cast<std::int64_t>(std::thread::hardware_concurrency());

    const auto num_threads =
        std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);

    ParallelForILoop(
        0, max_num_models,
        [&xs, &us, &txs, &tus, &jvps](auto j) {
          jvps.col(j) = Model::ModelPushfoward(xs.col(j), us.col(j), txs.col(j),
                                               tus.col(j));
        },
        num_threads);
  } else {
    for (auto j = 0; j != max_num_models; ++j) {
      jvps.col(j) =
          Model::ModelPushfoward(xs.col(j), us.col(j), txs.col(j), tus.col(j));
    }
  }
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_quadrotor_dynamics_f32"] =
      EncapsulateFunction(EvaluateDynamics<float>);
  dict["cpu_quadrotor_dynamics_f64"] =
      EncapsulateFunction(EvaluateDynamics<double>);
  dict["cpu_quadrotor_pushforward_f32"] =
      EncapsulateFunction(EvaluatePushforward<float>);
  dict["cpu_quadrotor_pushforward_f64"] =
      EncapsulateFunction(EvaluatePushforward<double>);
  return dict;
}

PYBIND11_MODULE(cpu_ops, m) { m.def("registrations", &Registrations); }

}  // namespace
