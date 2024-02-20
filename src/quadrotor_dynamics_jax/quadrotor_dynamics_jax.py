# -*- coding: utf-8 -*-

__all__ = ["quadrotor_dynamics"]

import functools

import jax
import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib import hlo_helpers

# Register the CPU XLA custom calls
from . import cpu_ops

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")

# If the GPU version exists, also register those
try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
else:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

# This function exposes the primitive to user code and this is the only
# public-facing function in this module


def quadrotor_dynamics(x, u):
    return _primary.bind(x, u)


STATE_SIZE = 10
INPUT_SIZE = 4

# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _quadrotor_dynamics_abstract(x, u):
    n_x, size_x = x.shape if len(x.shape) > 1 else (1,) + x.shape
    if size_x != STATE_SIZE:
        raise ValueError(f"Incorrect state dimension: {size_x} != {STATE_SIZE}")

    n_u, size_u = u.shape if len(u.shape) > 1 else (1,) + u.shape
    if size_u != INPUT_SIZE:
        raise ValueError(f"Incorrect input dimension: {size_u} != {INPUT_SIZE}")

    if n_x != n_u:
        raise ValueError(f"Mismatch in number of states and inputs {n_x} != {n_u}")

    dtype = dtypes.canonicalize_dtype(x.dtype)
    assert dtypes.canonicalize_dtype(u.dtype) == dtype
    return ShapedArray(x.shape, dtype)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
# This provides a mechanism for exposing our custom C++ and/or CUDA interfaces
# to the JAX XLA backend. We're wrapping two translation rules into one here:
# one for the CPU and one for the GPU
def _quadrotor_dynamics_lowering(ctx, x, u, platform="cpu", *args, **kw):

    # Checking that input types and shape agree
    if x.type.element_type != u.type.element_type:
        raise TypeError(
            f"Type mismatch: {x.type.element_type} != {u.type.element_type}"
        )

    # Extract the numpy type of the inputs
    mean_anom_aval, _ = ctx.avals_in
    np_dtype = np.dtype(mean_anom_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(x.type)
    dims = dtype.shape
    ndims = len(dims)
    layout = tuple(range(ndims - 1, -1, -1))

    # The total size of the input is the product across dimensions
    size = dims[0] if ndims > 1 else 1

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_quadrotor_dynamics_f32"
    elif np_dtype == np.float64:
        op_name = platform + "_quadrotor_dynamics_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    custom_call = functools.partial(
        hlo_helpers.custom_call,
        op_name,
        result_types=[dtype],
        result_layouts=[layout],
    )

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return custom_call(
            operands=[mlir.ir_constant(size), x, u],
            operand_layouts=[(), layout, layout],
        ).results

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'quadrotor_dynamics_jax' module was not compiled with CUDA support"
            )
        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_descriptor(size)

        return custom_call(
            operands=[x, u],
            operand_layouts=[layout, layout],
            backend_config=opaque,
        ).results

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************


def quaternion_product(lhs, rhs):
    return jnp.array(
        [
            lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
            lhs[3] * rhs[1] + lhs[1] * rhs[3] + lhs[2] * rhs[0] - lhs[0] * rhs[2],
            lhs[3] * rhs[2] + lhs[2] * rhs[3] + lhs[0] * rhs[1] - lhs[1] * rhs[0],
            lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2],
        ]
    )


def quaternion_rotate_point(quaternion, point):
    vec = quaternion[0:3]
    uv = jnp.cross(vec, point)
    uv += uv
    return point + quaternion[3] * uv + jnp.cross(vec, uv)


# In this case we don't need to define a transpose rule in order to support
# reverse and higher order differentiation. This might not be true in other
# applications, so check out the "How JAX primitives work" tutorial in the JAX
# documentation for more info as necessary.
def _quadrotor_dynamics_jvp(args, tangents):
    # mean_anom, ecc = args
    # d_mean_anom, d_ecc = tangents
    x, u = args
    tx, tu = tangents

    # # We use "bind" here because we don't want to mod the mean anomaly again
    primals = _primary.bind(*args)

    q = x[3:7]
    f = u[0]
    omega_q = jnp.append(u[1:4] / 2.0, 0.0)

    qv_x_i3 = jnp.array([q[1], -q[0], 0.0])
    i3_x_tx_35 = jnp.array([-tx[4], tx[3], 0.0])

    ux_13_q = jnp.append(tu[1:4] / 2.0, 0.0)
    df_tx = jnp.concatenate(
        [
            tx[7:10],
            quaternion_product(tx[3:7], omega_q) + quaternion_product(q, ux_13_q),
            2.0
            * f
            * (
                q[2] * tx[3:6]
                - jnp.cross(qv_x_i3, tx[3:6])
                - q[3] * i3_x_tx_35
                + (qv_x_i3 + jnp.array([0.0, 0.0, q[3]])) * tx[6]
            )
            + quaternion_rotate_point(q, jnp.array([0.0, 0.0, 1.0])) * tu[0],
        ]
    )

    return primals, df_tx


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************


# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _quadrotor_dynamics_batch(args, axes):
    assert axes[0] == axes[1]
    return quadrotor_dynamics(*args), axes[0]


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_primary = core.Primitive("quadrotor_dynamics")
_primary.multiple_results = False
_primary.def_impl(functools.partial(xla.apply_primitive, _primary))
_primary.def_abstract_eval(_quadrotor_dynamics_abstract)

# Connect the XLA translation rules for JIT compilation
for platform in ["cpu", "gpu"]:
    mlir.register_lowering(
        _primary,
        functools.partial(_quadrotor_dynamics_lowering, platform=platform),
        platform=platform,
    )

# Connect the JVP and batching rules
ad.primitive_jvps[_primary] = _quadrotor_dynamics_jvp
batching.primitive_batchers[_primary] = _quadrotor_dynamics_batch
