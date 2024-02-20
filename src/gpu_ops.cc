// This file defines the Python interface to the XLA custom call implemented on
// the GPU. Like in cpu_ops.cc, we export a separate capsule for each supported
// dtype, but we also include one extra method
// "build_quadrotor_dynamics_descriptor" to generate an opaque representation of
// the problem size that will be passed to the op. The actually implementation
// of the custom call can be found in kernels.cc.cu.

#include "extending_jax/kernels.h"
#include "extending_jax/pybind11_kernel_helpers.h"

namespace model {

namespace {
pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["gpu_quadrotor_dynamics_f32"] =
      EncapsulateFunction(gpu_quadrotor_dynamics_f32);
  dict["gpu_quadrotor_dynamics_f64"] =
      EncapsulateFunction(gpu_quadrotor_dynamics_f64);
  dict["gpu_quadrotor_pushforward_f32"] =
      EncapsulateFunction(gpu_quadrotor_pushforward_f32);
  dict["gpu_quadrotor_pushforward_f64"] =
      EncapsulateFunction(gpu_quadrotor_pushforward_f64);

  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("registrations", &Registrations);
  m.def("build_descriptor",
        [](std::int64_t size) { return PackDescriptor(Descriptor{size}); });
}
}  // namespace

}  // namespace model
